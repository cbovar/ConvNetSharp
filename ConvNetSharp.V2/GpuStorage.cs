using System.Numerics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaDNN;
using Math = System.Math;

namespace ConvNetSharp.V2;

/// <summary>
///     Represents a GPU-based storage for volumes in ConvNetSharp.
///     This class manages data storage, transfers between host and device,
///     and GPU-specific operations like addition and memory clearing.
/// </summary>
/// <typeparam name="T">The data type of the storage, must be a struct.</typeparam>
public class GpuStorage<T> : IVolumeStorage<T>, IDisposable where T : struct, INumber<T>
{
    private readonly CudaDeviceVariable<T> _deviceData;
    private readonly T[] _hostData;
    private bool _isDisposed;

    /// <summary>
    ///     Initializes a new instance of the <see cref="GpuStorage{T}" /> class.
    ///     Allocates GPU memory for the given data and sets the initial data location to the host.
    /// </summary>
    /// <param name="data">An optional array containing the initial data. If null, an empty array will be allocated.</param>
    /// <param name="shape">The shape of the tensor represented by this storage.</param>
    public GpuStorage(T[] data, Shape shape)
    {
        // Ensure GPU context is initialized
        if (!GpuContext.Instance.IsInitialized)
        {
            GpuContext.Instance.Initialize();
        }

        this.Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        this._hostData = data ?? new T[shape.TotalLength];
        this._deviceData = new CudaDeviceVariable<T>(shape.TotalLength);
        this.DataLocation = DataLocation.Host; // Initially, data is only on the host
    }

    /// <summary>
    ///     Initializes a new instance of the <see cref="GpuStorage{T}" /> class with a specified shape.
    /// </summary>
    /// <param name="shape">The shape of the tensor represented by this storage.</param>
    public GpuStorage(Shape shape)
        : this(null, shape) // Delegate to the other constructor
    {
    }

    /// <summary>
    ///     Gets or sets the location where the valid data resides (Host or Device).
    /// </summary>
    public DataLocation DataLocation { get; private set; } // Tracks where the valid data is


    /// <summary>
    ///     Exposes the device pointer for this storage.
    /// </summary>
    public CUdeviceptr DevicePointer => this._deviceData.DevicePointer;

    /// <summary>
    ///     Releases resources used by the storage.
    /// </summary>
    public void Dispose()
    {
        if (this._isDisposed)
        {
            return;
        }

        this._deviceData.Dispose();
        this._isDisposed = true;
    }

    /// <summary>
    ///     Gets the device type where the data is stored (e.g., CPU or GPU).
    /// </summary>
    public Device Device => Device.GPU;

    /// <summary>
    ///     Gets the shape of the volume.
    /// </summary>
    public Shape Shape { get; }

    /// <summary>
    ///     Gets the data from the storage. If the data is on the device, it synchronizes back to the host.
    /// </summary>
    /// <returns>The data array.</returns>
    public T[] GetData()
    {
        this.EnsureNotDisposed(); // Ensure the object is not disposed

        if (this.DataLocation == DataLocation.Device)
        {
            // Sync device data back to host
            this._deviceData.CopyToHost(this._hostData);
            this.DataLocation = DataLocation.Host;
        }

        return this._hostData;
    }

    /// <summary>
    ///     Sets the entire data array of the volume.
    /// </summary>
    /// <param name="data">An array containing the data to be set.</param>
    /// <exception cref="ArgumentException">
    ///     Thrown when the length of the provided data does not match the total length of the volume.
    /// </exception>
    public void SetData(T[] data)
    {
        this.EnsureNotDisposed();

        if (data == null)
        {
            throw new ArgumentNullException(nameof(data), "Data cannot be null.");
        }

        if (data.Length != this.Shape.TotalLength)
        {
            throw new ArgumentException(
                $"The length of the provided data ({data.Length}) does not match the total length of the volume ({this.Shape.TotalLength}).");
        }

        // Copy the data to the host storage
        Array.Copy(data, this._hostData, data.Length);

        // Mark data as updated on the host
        this.DataLocation = DataLocation.Host;

        // Optionally, sync the data to the device immediately
        this.CopyToDevice();
    }

    /// <summary>
    ///     Applies a user-defined operation to the data.
    ///     ** SLOW **
    /// </summary>
    /// <param name="operation">The operation to apply to each element of the storage.</param>
    public void ApplyOperation(Func<T, T> operation)
    {
        for (var i = 0; i < this.Shape.TotalLength; i++)
        {
            this.Set(i, operation(this.Get(i)));
        }
    }

    /// <summary>
    ///     Adds the data from another storage to the current storage and places the result in a third storage.
    /// </summary>
    /// <param name="other">The other storage to add.</param>
    /// <param name="result">The storage where the result is stored.</param>
    public void Add(IVolumeStorage<T> other, IVolumeStorage<T> result)
    {
        if (other is not GpuStorage<T> gpuOther || result is not GpuStorage<T> gpuResult)
        {
            throw new ArgumentException("Both other and result must be of type GpuStorage<T>.");
        }

        // Validate broadcasting compatibility
        if ((!this.Shape.IsBroadcastableTo(gpuOther.Shape) && !gpuOther.Shape.IsBroadcastableTo(this.Shape)) ||
            !result.Shape.Equals(Shape.GetBroadcastedShape(this.Shape, gpuOther.Shape)))
        {
            Console.WriteLine("Broadcasting validation failed for GPU Add:");
            Console.WriteLine($"Left Shape: {this.Shape}");
            Console.WriteLine($"Right Shape: {gpuOther.Shape}");
            Console.WriteLine($"Result Shape: {result.Shape}");
            Console.WriteLine($"Broadcasted Shape: {Shape.GetBroadcastedShape(this.Shape, gpuOther.Shape)}");
            throw new ArgumentException("Shapes of storages are not broadcastable.");
        }

        if (ReferenceEquals(other, result))
        {
            // In-place addition: Add this tensor directly to result
            this.AddToResult(result, T.One);
        }
        else
        {
            // Standard case: clear result, add this and other tensors
            result.Clear(); // Optional, as beta defaults to T.Zero
            this.AddToResult(result, T.Zero); // Add `this` to an empty result
            gpuOther.AddToResult(result, T.One); // Add `other` to result
        }
    }


    /// <summary>
    ///     Accumulates the data from the current storage into another storage in-place.
    /// </summary>
    /// <param name="result">The storage where the data is accumulated.</param>
    /// <exception cref="ArgumentException">Thrown if the storage types or shapes do not match.</exception>
    public void Add(IVolumeStorage<T> result)
    {
        // In-place addition: Use beta = T.One to retain existing data in result
        this.AddToResult(result, T.One);
    }

    /// <summary>
    ///     Clears the storage data by setting all values to zero on the GPU.
    /// </summary>
    public void Clear()
    {
        this.EnsureNotDisposed();

        // Clear device memory
        var elementSize = Marshal.SizeOf<T>();
        var result = DriverAPINativeMethods.Memset.cuMemsetD32_v2(
            this._deviceData.DevicePointer,
            0,
            this._deviceData.Size * elementSize / sizeof(int) // Element count conversion
        );

        if (result != CUResult.Success)
        {
            throw new CudaException(result);
        }

        // Mark data as being on the device
        this.DataLocation = DataLocation.Device;
    }

    /// <summary>
    ///     Performs a convolution operation on the input volume using the specified filter.
    /// </summary>
    /// <param name="filterStorage">The filter volume to use for convolution.</param>
    /// <param name="xpad">The padding size along the width dimension.</param>
    /// <param name="ypad">The padding size along the height dimension.</param>
    /// <param name="stride">The stride size for the convolution.</param>
    /// <param name="outputStorage">The volume to store the convolution result.</param>
    public void Convolution(IVolumeStorage<T> filterStorage, int xpad, int ypad, int stride,
        IVolumeStorage<T> outputStorage)
    {
        if (filterStorage is not GpuStorage<T> gpuFilterStorage || outputStorage is not GpuStorage<T> gpuOutputStorage)
        {
            throw new ArgumentException("Both filterStorage and outputStorage must be of type GpuStorage<T>.");
        }

        if (this.Shape.Dimensions[2] != filterStorage.Shape.Dimensions[2])
        {
            throw new ArgumentException("Input depth and filter depth must match.");
        }

        // Ensure all storages are on the device
        this.CopyToDevice();
        gpuFilterStorage.CopyToDevice();
        gpuOutputStorage.CopyToDevice();

        using var dataDesc = new TensorDescriptor();
        using var filterDesc = new FilterDescriptor();
        using var outputDesc = new TensorDescriptor();
        using var convolutionDesc = new ConvolutionDescriptor();

        // Define the convolution descriptor
        convolutionDesc.SetConvolution2dDescriptor(ypad, xpad, stride, stride, 1, 1,
            cudnnConvolutionMode.CrossCorrelation, GetCudnnDataType());

        // Set the tensor descriptors
        dataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, GetCudnnDataType(),
            this.Shape.Dimensions[3], // Batch size
            this.Shape.Dimensions[2], // Input channels
            this.Shape.Dimensions[1], // Input height
            this.Shape.Dimensions[0]); // Input width

        filterDesc.SetFilter4dDescriptor(GetCudnnDataType(), cudnnTensorFormat.NCHW,
            filterStorage.Shape.Dimensions[3], // Output channels
            filterStorage.Shape.Dimensions[2], // Filter depth
            filterStorage.Shape.Dimensions[1], // Filter height
            filterStorage.Shape.Dimensions[0]); // Filter width

        outputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, GetCudnnDataType(),
            outputStorage.Shape.Dimensions[3], // Batch size
            outputStorage.Shape.Dimensions[2], // Output channels
            outputStorage.Shape.Dimensions[1], // Output height
            outputStorage.Shape.Dimensions[0]); // Output width

        // Get the convolution algorithm
        var algo = GpuContext.Instance.CudnnContext.GetConvolutionForwardAlgorithm(
            dataDesc, filterDesc, convolutionDesc, outputDesc, cudnnConvolutionFwdPreference.PreferFastest,
            IntPtr.Zero);

        // Allocate workspace
        var workspaceSize = GpuContext.Instance.CudnnContext.GetConvolutionForwardWorkspaceSize(
            dataDesc, filterDesc, convolutionDesc, outputDesc, algo);

        workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

        using var workspace = new CudaDeviceVariable<byte>(workspaceSize);

        // Perform the convolution
        if (typeof(T) == typeof(float))
        {
            GpuContext.Instance.CudnnContext.ConvolutionForward(
                1.0f, // Alpha
                dataDesc, this._deviceData as CudaDeviceVariable<float>,
                filterDesc, gpuFilterStorage._deviceData as CudaDeviceVariable<float>,
                convolutionDesc, algo, workspace,
                0.0f, // Beta
                outputDesc, gpuOutputStorage._deviceData as CudaDeviceVariable<float>);
        }
        else if (typeof(T) == typeof(double))
        {
            GpuContext.Instance.CudnnContext.ConvolutionForward(
                1.0, // Alpha
                dataDesc, this._deviceData as CudaDeviceVariable<double>,
                filterDesc, gpuFilterStorage._deviceData as CudaDeviceVariable<double>,
                convolutionDesc, algo, workspace,
                0.0, // Beta
                outputDesc, gpuOutputStorage._deviceData as CudaDeviceVariable<double>);
        }
        else
        {
            throw new NotSupportedException($"Data type {typeof(T)} is not supported for convolution.");
        }

        // Mark the output as being on the device
        gpuOutputStorage.DataLocation = DataLocation.Device;
    }

    /// <summary>
    ///     Repeats the elements of this volume according to the specified replication factors,
    ///     and stores the result in the provided result volume.
    /// </summary>
    /// <param name="reps">The replication factors volume.</param>
    /// <param name="result">The volume to store the tiled output.</param>
    public void Tile(IVolumeStorage<T> reps, IVolumeStorage<T> result)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///     Multiplies all elements of this volume by a scalar value and stores the result in the provided GPU storage.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <param name="resultStorage">The GPU storage where the result of the operation will be stored.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="resultStorage" /> is null.</exception>
    /// <exception cref="ArgumentException">
    ///     Thrown when <paramref name="resultStorage" /> is not of type
    ///     <see cref="GpuStorage{T}" />.
    /// </exception>
    /// <exception cref="NotSupportedException">Thrown when the type <typeparamref name="T" /> is not supported.</exception>
    public void MultiplyByScalar(T scalar, IVolumeStorage<T> resultStorage)
    {
        if (resultStorage == null)
        {
            throw new ArgumentNullException(nameof(resultStorage), "Result storage cannot be null.");
        }

        if (resultStorage is not GpuStorage<T> gpuResultStorage)
        {
            throw new ArgumentException("Result storage must be of type GpuStorage<T>.", nameof(resultStorage));
        }

        // Ensure the data is on the device
        this.CopyToDevice();

        // If resultStorage is not the same as this, copy the data to resultStorage
        if (!ReferenceEquals(this, gpuResultStorage))
        {
            gpuResultStorage.CopyFrom(this);
        }

        // Perform scaling based on the type of T
        if (typeof(T) == typeof(float))
        {
            GpuContext.Instance.CublasContext.Scale(
                Convert.ToSingle(scalar),
                gpuResultStorage._deviceData as CudaDeviceVariable<float>,
                1);
        }
        else if (typeof(T) == typeof(double))
        {
            GpuContext.Instance.CublasContext.Scale(
                Convert.ToDouble(scalar),
                gpuResultStorage._deviceData as CudaDeviceVariable<double>,
                1);
        }
        else
        {
            throw new NotSupportedException($"Type {typeof(T)} is not supported for GPU operations.");
        }
    }

    /// <summary>
    ///     Subtracts the elements of another volume from this volume, storing the result in a third volume.
    /// </summary>
    /// <param name="rightStorage">The volume to subtract from this volume.</param>
    /// <param name="resultStorage">The volume to store the result.</param>
    public void Subtract(IVolumeStorage<T> rightStorage, IVolumeStorage<T> resultStorage)
    {
        this.Op(rightStorage, resultStorage, cudnnOpTensorOp.OpTensorAdd);
    }

    public void MatMultiply(IVolumeStorage<T> rightStorage, IVolumeStorage<T> resultStorage)
    {
        if (rightStorage is not GpuStorage<T> gpuRightStorage || resultStorage is not GpuStorage<T> gpuResultStorage)
        {
            throw new ArgumentException("Both rightStorage and resultStorage must be of type GpuStorage<T>.");
        }

        // Copy data to device memory
        this.CopyToDevice();
        gpuRightStorage.CopyToDevice();
        gpuResultStorage.CopyToDevice();

        var leftShape = this.Shape; // [columns, rows, channels, batch]
        var rightShape = gpuRightStorage.Shape;
        var resultShape = gpuResultStorage.Shape;

        // Validate matrix dimensions for multiplication
        if (leftShape.Dimensions[0] != rightShape.Dimensions[1]) // Columns of A != Rows of B
        {
            throw new ArgumentException(
                $"Matrix dimensions do not align for multiplication. Left: {this.Shape}, Right: {gpuRightStorage.Shape}");
        }

        var batchSize = Math.Max(leftShape.Dimensions[3], rightShape.Dimensions[3]);
        if (batchSize != resultShape.Dimensions[3])
        {
            throw new ArgumentException("Batch size mismatch between operands and result.");
        }

        var m = leftShape.Dimensions[1]; // Rows in left matrix (A)
        var n = rightShape.Dimensions[0]; // Columns in right matrix (B)
        var k = leftShape.Dimensions[0]; // Shared dimension

        var broadCastLeft = this.Shape.Dimensions[3] == 1;
        var broadCastRight = rightShape.Dimensions[3] == 1;

        for (var b = 0; b < resultShape.Dimensions[3]; b++) // for each batch
        {
            if (typeof(T) == typeof(double))
            {
                GpuContext.Instance.CublasContext.Gemm(Operation.NonTranspose, Operation.NonTranspose,
                    n, m, k,
                    1.0,
                    broadCastRight
                        ? gpuRightStorage._deviceData as CudaDeviceVariable<double>
                        : new CudaDeviceVariable<double>(gpuRightStorage._deviceData.DevicePointer +
                                                         b * k * n * sizeof(double)), n,
                    broadCastLeft
                        ? this._deviceData as CudaDeviceVariable<double>
                        : new CudaDeviceVariable<double>(
                            this._deviceData.DevicePointer + b * k * m * sizeof(double)), k,
                    0.0,
                    new CudaDeviceVariable<double>(gpuResultStorage._deviceData.DevicePointer +
                                                   b * n * m * sizeof(double)),
                    n);
            }
            else if (typeof(T) == typeof(float))
            {
                GpuContext.Instance.CublasContext.Gemm(Operation.NonTranspose, Operation.NonTranspose,
                    n, m, k,
                    1.0f,
                    broadCastRight
                        ? gpuRightStorage._deviceData as CudaDeviceVariable<float>
                        : new CudaDeviceVariable<float>(gpuRightStorage._deviceData.DevicePointer +
                                                        b * k * n * sizeof(float)), n,
                    broadCastLeft
                        ? this._deviceData as CudaDeviceVariable<float>
                        : new CudaDeviceVariable<float>(
                            this._deviceData.DevicePointer + b * k * m * sizeof(float)), k,
                    0.0f,
                    new CudaDeviceVariable<float>(gpuResultStorage._deviceData.DevicePointer +
                                                  b * n * m * sizeof(float)),
                    n);
            }
        }

        gpuResultStorage.DataLocation = DataLocation.Device;
    }

    /// <summary>
    ///     Performs element-wise multiplication between the current volume storage and another volume storage,
    ///     and stores the result in the provided result storage.
    /// </summary>
    /// <param name="rightStorage">The right-hand operand volume storage for the element-wise multiplication.</param>
    /// <param name="resultStorage">The volume storage where the result of the element-wise multiplication will be stored.</param>
    /// <exception cref="ArgumentNullException">
    ///     Thrown when <paramref name="rightStorage" /> or
    ///     <paramref name="resultStorage" /> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    ///     Thrown when the shapes of the volume storages are not compatible for element-wise
    ///     multiplication.
    /// </exception>
    /// <remarks>
    ///     The method computes the operation:
    ///     <c>result[i] = this[i] * rightStorage[i]</c>
    ///     for each element in the volumes. The shapes of the volumes must match or be broadcastable for the operation to
    ///     succeed.
    /// </remarks>
    public void Multiply(IVolumeStorage<T> rightStorage, IVolumeStorage<T> resultStorage)
    {
        // Perform element-wise multiplication using cuDNN's tensor operations
        this.Op(rightStorage, resultStorage, cudnnOpTensorOp.OpTensorMul);
    }

    /// <summary>
    ///     Computes the transpose of the current volume storage and stores it in the provided result storage.
    /// </summary>
    /// <param name="resultStorage">The volume storage where the transposed data will be stored.</param>
    /// <exception cref="ArgumentNullException">
    ///     Thrown when <paramref name="resultStorage" /> is null.
    /// </exception>
    public void Transpose(IVolumeStorage<T> resultStorage)
    {
        if (resultStorage is not GpuStorage<T> gpuResultStorage)
        {
            throw new ArgumentException("Result storage must be of type GpuStorage<T>.");
        }

        // Ensure data is on the device
        this.CopyToDevice();
        gpuResultStorage.CopyToDevice();

        var m = this.Shape.Dimensions[1]; // Rows of the original matrix
        var n = this.Shape.Dimensions[0]; // Columns of the original matrix

        // Validate the result storage shape
        var expectedShape = new Shape(this.Shape.Dimensions[1], this.Shape.Dimensions[0], this.Shape.Dimensions[2],
            this.Shape.Dimensions[3]);
        if (!gpuResultStorage.Shape.Equals(expectedShape))
        {
            throw new ArgumentException(
                $"Result storage shape is invalid for transposition. Expected {expectedShape}, got {gpuResultStorage.Shape}.");
        }

        for (var batch = 0; batch < this.Shape.Dimensions[3]; batch++)
        {
            var inputOffset = this._deviceData.DevicePointer + batch * m * n * Marshal.SizeOf<T>();
            var resultOffset = gpuResultStorage._deviceData.DevicePointer + batch * n * m * Marshal.SizeOf<T>();

            if (typeof(T) == typeof(double))
            {
                var handle = GpuContext.Instance.CublasContext;
                handle.Geam(
                    Operation.Transpose,
                    Operation.NonTranspose, // B is ignored with beta = 0
                    m, // Rows of transposed matrix (original columns)
                    n, // Columns of transposed matrix (original rows)
                    1.0, // Alpha
                    new CudaDeviceVariable<double>(inputOffset), // A
                    n, // Leading dimension of A
                    new CudaDeviceVariable<double>(inputOffset), // Use A as B
                    m, // Leading dimension of B
                    0.0, // Beta (ignore B)
                    new CudaDeviceVariable<double>(resultOffset), // Result matrix C
                    m // Leading dimension of result
                );
            }
            else if (typeof(T) == typeof(float))
            {
                var handle = GpuContext.Instance.CublasContext;
                handle.Geam(
                    Operation.Transpose,
                    Operation.NonTranspose, // B is ignored with beta = 0
                    m, // Rows of transposed matrix (original columns)
                    n, // Columns of transposed matrix (original rows)
                    1.0f, // Alpha
                    new CudaDeviceVariable<float>(inputOffset), // A
                    n, // Leading dimension of A
                    new CudaDeviceVariable<float>(inputOffset), // Use A as B
                    m, // Leading dimension of B
                    0.0f, // Beta (ignore B)
                    new CudaDeviceVariable<float>(resultOffset), // Result matrix C
                    m // Leading dimension of result
                );
            }
            else
            {
                throw new NotSupportedException($"Data type {typeof(T)} is not supported for transposition.");
            }
        }

        gpuResultStorage.DataLocation = DataLocation.Device;
    }

    /// <summary>
    ///     Computes the element-wise power of the current volume with the specified exponent volume,
    ///     and stores the result in the provided result storage.
    /// </summary>
    /// <param name="power">The volume storage containing the exponents for the power operation.</param>
    /// <param name="result">The volume storage where the result of the power operation will be stored.</param>
    /// <exception cref="ArgumentNullException">
    ///     Thrown when <paramref name="power" /> or <paramref name="result" /> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    ///     Thrown when the shapes of the current volume, <paramref name="power" />, and
    ///     <paramref name="result" /> are not compatible for the element-wise power operation.
    /// </exception>
    /// <remarks>
    ///     The method computes the operation:
    ///     <c>result[i] = this[i] ^ power[i]</c>
    ///     for each element in the volumes. The shapes of the volumes must match or be broadcastable for the operation to
    ///     succeed.
    /// </remarks>
    public void Power(IVolumeStorage<T> power, IVolumeStorage<T> result)
    {
        if (power is not GpuStorage<T> powerStorage || result is not GpuStorage<T> resultStorage)
        {
            throw new ArgumentException("Both power and result must be of type GpuStorage<T>.");
        }

        // Validate shapes
        if (!this.Shape.Equals(power.Shape) && power.Shape.TotalLength > 1)
        {
            throw new ArgumentException(
                "Base volume and power volume should have the same shape, OR power should be a scalar.");
        }

        if (!this.Shape.Equals(result.Shape))
        {
            throw new ArgumentException(
                $"Base volume and result volume should have the same shape ({this.Shape} != {result.Shape}).");
        }

        // Determine if the power volume is a scalar
        var rightIsScalar = power.Shape.TotalLength == 1 ? 1 : 0;

        // Launch the kernel
        var kernelLoader = new KernelLoader<T>(GpuContext.Instance);
        var kernelName = "PowerKernel";
        var kernelCode = this.GetKernelCode(typeof(T));

        kernelLoader.LoadKernel(kernelName, kernelCode);
        kernelLoader.RunKernel(kernelName, this, powerStorage, resultStorage, rightIsScalar);

        Console.WriteLine("Power operation completed successfully.");
    }

    /// <summary>
    ///     Retrieves a single value from the storage by its flat index.
    /// </summary>
    /// <param name="i">The flat index of the desired element.</param>
    /// <returns>The value stored at the specified index.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the storage has been disposed.</exception>
    /// <remarks>This method assumes the storage is represented in a 1D flattened format.</remarks>
    public T Get(int i)
    {
        this.EnsureNotDisposed();
        this.EnsureHostData();
        return this._hostData[i];
    }

    /// <summary>
    ///     Retrieves a single value from the storage based on 2D coordinates (width and height).
    /// </summary>
    /// <param name="w">The width index (horizontal position).</param>
    /// <param name="h">The height index (vertical position).</param>
    /// <returns>The value stored at the specified 2D coordinate.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the storage has been disposed.</exception>
    /// <exception cref="ArgumentException">Thrown if the coordinates are out of bounds.</exception>
    /// <remarks>Assumes the storage has at least two dimensions (width and height).</remarks>
    public T Get(int w, int h)
    {
        this.EnsureNotDisposed();
        this.EnsureHostData();

        var index = h * this.Shape.Dimensions[0] + w;
        return this._hostData[index];
    }

    /// <summary>
    ///     Retrieves a single value from the storage based on 3D coordinates (width, height, and channel).
    /// </summary>
    /// <param name="w">The width index (horizontal position).</param>
    /// <param name="h">The height index (vertical position).</param>
    /// <param name="c">The channel index (depth or feature map).</param>
    /// <returns>The value stored at the specified 3D coordinate.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the storage has been disposed.</exception>
    /// <exception cref="ArgumentException">Thrown if the coordinates are out of bounds.</exception>
    /// <remarks>Assumes the storage has at least three dimensions (width, height, and channel).</remarks>
    public T Get(int w, int h, int c)
    {
        this.EnsureNotDisposed();
        this.EnsureHostData();

        var index = c * this.Shape.Dimensions[0] * this.Shape.Dimensions[1]
                    + h * this.Shape.Dimensions[0] + w;
        return this._hostData[index];
    }

    /// <summary>
    ///     Retrieves a single value from the storage based on 4D coordinates (width, height, channel, and batch).
    /// </summary>
    /// <param name="w">The width index (horizontal position).</param>
    /// <param name="h">The height index (vertical position).</param>
    /// <param name="c">The channel index (depth or feature map).</param>
    /// <param name="n">The batch index (sample in a batch).</param>
    /// <returns>The value stored at the specified 4D coordinate.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the storage has been disposed.</exception>
    /// <exception cref="ArgumentException">Thrown if the coordinates are out of bounds.</exception>
    /// <remarks>Assumes the storage has at least four dimensions (width, height, channel, and batch).</remarks>
    public T Get(int w, int h, int c, int n)
    {
        this.EnsureNotDisposed();
        this.EnsureHostData();

        var index = n * this.Shape.Dimensions[0] * this.Shape.Dimensions[1] * this.Shape.Dimensions[2]
                    + c * this.Shape.Dimensions[0] * this.Shape.Dimensions[1]
                    + h * this.Shape.Dimensions[0] + w;
        return this._hostData[index];
    }

    /// <summary>
    ///     Retrieves a single value from the storage based on an n-dimensional coordinate array.
    /// </summary>
    /// <param name="coordinates">An array of indices specifying the desired element's position in each dimension.</param>
    /// <returns>The value stored at the specified n-dimensional coordinate.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the storage has been disposed.</exception>
    /// <exception cref="ArgumentException">
    ///     Thrown if the number of coordinates does not match the number of dimensions in the
    ///     storage shape.
    /// </exception>
    /// <remarks>
    ///     This method dynamically calculates the flat index using strides, making it suitable for arbitrary-dimensional
    ///     storage.
    /// </remarks>
    public T Get(int[] coordinates)
    {
        this.EnsureNotDisposed();
        this.EnsureHostData();

        if (coordinates.Length != this.Shape.Dimensions.Length)
        {
            throw new ArgumentException("Coordinate dimensions do not match the storage shape.");
        }

        var index = 0;
        var stride = 1;

        for (var i = this.Shape.Dimensions.Length - 1; i >= 0; i--)
        {
            index += coordinates[i] * stride;
            stride *= this.Shape.Dimensions[i];
        }

        return this._hostData[index];
    }

    /// <summary>
    ///     Sets a single value in the storage at the specified flat index.
    /// </summary>
    /// <param name="i">The flat index where the value will be set.</param>
    /// <param name="value">The value to set at the specified index.</param>
    public void Set(int i, T value)
    {
        this.EnsureNotDisposed();
        this.EnsureHostData();

        if (i < 0 || i >= this._hostData.Length)
        {
            throw new IndexOutOfRangeException("Index out of bounds.");
        }

        this._hostData[i] = value;
        this.DataLocation = DataLocation.Host; // Mark data as updated on the host
    }

    /// <summary>
    ///     Sets a single value in the storage based on 2D coordinates (width and height).
    /// </summary>
    /// <param name="w">The width index (horizontal position).</param>
    /// <param name="h">The height index (vertical position).</param>
    /// <param name="value">The value to set at the specified 2D coordinate.</param>
    public void Set(int w, int h, T value)
    {
        var index = h * this.Shape.Dimensions[0] + w;
        this.Set(index, value);
    }

    /// <summary>
    ///     Sets a single value in the storage based on 3D coordinates (width, height, and channel).
    /// </summary>
    /// <param name="w">The width index (horizontal position).</param>
    /// <param name="h">The height index (vertical position).</param>
    /// <param name="c">The channel index (depth or feature map).</param>
    /// <param name="value">The value to set at the specified 3D coordinate.</param>
    public void Set(int w, int h, int c, T value)
    {
        var index = c * this.Shape.Dimensions[0] * this.Shape.Dimensions[1]
                    + h * this.Shape.Dimensions[0]
                    + w;
        this.Set(index, value);
    }

    /// <summary>
    ///     Sets a single value in the storage based on 4D coordinates (width, height, channel, and batch).
    /// </summary>
    /// <param name="w">The width index (horizontal position).</param>
    /// <param name="h">The height index (vertical position).</param>
    /// <param name="c">The channel index (depth or feature map).</param>
    /// <param name="n">The batch index (sample in a batch).</param>
    /// <param name="value">The value to set at the specified 4D coordinate.</param>
    public void Set(int w, int h, int c, int n, T value)
    {
        var index = n * this.Shape.Dimensions[0] * this.Shape.Dimensions[1] * this.Shape.Dimensions[2]
                    + c * this.Shape.Dimensions[0] * this.Shape.Dimensions[1]
                    + h * this.Shape.Dimensions[0]
                    + w;
        this.Set(index, value);
    }

    /// <summary>
    ///     Sets a single value in the storage based on an n-dimensional coordinate array.
    /// </summary>
    /// <param name="coordinates">An array of indices specifying the position in each dimension.</param>
    /// <param name="value">The value to set at the specified n-dimensional coordinate.</param>
    public void Set(int[] coordinates, T value)
    {
        if (coordinates.Length != this.Shape.Dimensions.Length)
        {
            throw new ArgumentException("Coordinate dimensions do not match the storage shape.");
        }

        var index = 0;
        var stride = 1;

        for (var i = this.Shape.Dimensions.Length - 1; i >= 0; i--)
        {
            index += coordinates[i] * stride;
            stride *= this.Shape.Dimensions[i];
        }

        this.Set(index, value);
    }

    private string GetKernelCode(Type type)
    {
        var floatType = type == typeof(float) ? "float" : "double";
        var powFunc = type == typeof(float) ? "powf" : "pow";

        return $@"
    extern ""C"" __global__ void PowerKernel(int n, const {floatType}* left, const {floatType}* right, {floatType}* output, int rightIsScalar)
    {{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {{
            if (rightIsScalar == 1)
            {{
                output[i] = {powFunc}(left[i], right[0]);
            }}
            else
            {{
                output[i] = {powFunc}(left[i], right[i]);
            }}
        }}
    }}";
    }

    public void Op(IVolumeStorage<T> rightStorage, IVolumeStorage<T> resultStorage, cudnnOpTensorOp op)
    {
        if (rightStorage is not GpuStorage<T> gpuRightStorage || resultStorage is not GpuStorage<T> gpuResultStorage)
        {
            throw new ArgumentException("Both rightStorage and resultStorage must be of type GpuStorage<T>.");
        }

        // Ensure data is on the device
        this.CopyToDevice();
        gpuRightStorage.CopyToDevice();
        gpuResultStorage.CopyToDevice();

        // Validate tensor shapes
        if (!this.Shape.IsBroadcastableTo(gpuResultStorage.Shape) || !gpuRightStorage.Shape.IsBroadcastableTo(gpuResultStorage.Shape))
        {
            throw new ArgumentException("Shapes are not broadcastable.");
        }


        // Configure tensor descriptors
        using var leftDesc = new TensorDescriptor();
        using var rightDesc = new TensorDescriptor();
        using var resultDesc = new TensorDescriptor();
        using var opTensorDesc = new OpTensorDescriptor(GpuContext.Instance.CudnnContext);

        leftDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, GetCudnnDataType(),
            this.Shape.Dimensions[3], this.Shape.Dimensions[2], this.Shape.Dimensions[1], this.Shape.Dimensions[0]);

        rightDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, GetCudnnDataType(),
            gpuRightStorage.Shape.Dimensions[3], gpuRightStorage.Shape.Dimensions[2],
            gpuRightStorage.Shape.Dimensions[1], gpuRightStorage.Shape.Dimensions[0]);

        resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, GetCudnnDataType(),
            gpuResultStorage.Shape.Dimensions[3], gpuResultStorage.Shape.Dimensions[2],
            gpuResultStorage.Shape.Dimensions[1], gpuResultStorage.Shape.Dimensions[0]);

        // Configure the operation descriptor
        opTensorDesc.SetOpTensorDescriptor(op, GetCudnnDataType(), cudnnNanPropagation.PropagateNan);

        // Perform the operation
        if (typeof(T) == typeof(float))
        {
            var alpha = 1.0f;
            var beta = op == cudnnOpTensorOp.OpTensorMul ? 1.0f : -1.0f; // Adjust beta based on operation
            var zero = 0.0f;

            var status = CudaDNNNativeMethods.cudnnOpTensor(
                GpuContext.Instance.CudnnContext.Handle,
                opTensorDesc.Desc,
                ref alpha, leftDesc.Desc, this._deviceData.DevicePointer,
                ref beta, rightDesc.Desc, gpuRightStorage._deviceData.DevicePointer,
                ref zero, resultDesc.Desc, gpuResultStorage._deviceData.DevicePointer);

            if (status != cudnnStatus.Success)
            {
                throw new CudaException($"cudnnOpTensor failed with error: {status}");
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var alpha = 1.0;
            var beta = op == cudnnOpTensorOp.OpTensorMul ? 1.0 : -1.0; // Adjust beta based on operation
            var zero = 0.0;

            var status = CudaDNNNativeMethods.cudnnOpTensor(
                GpuContext.Instance.CudnnContext.Handle,
                opTensorDesc.Desc,
                ref alpha, leftDesc.Desc, this._deviceData.DevicePointer,
                ref beta, rightDesc.Desc, gpuRightStorage._deviceData.DevicePointer,
                ref zero, resultDesc.Desc, gpuResultStorage._deviceData.DevicePointer);

            if (status != cudnnStatus.Success)
            {
                throw new CudaException($"cudnnOpTensor failed with error: {status}");
            }
        }
        else
        {
            throw new NotSupportedException($"Data type {typeof(T)} is not supported for the operation.");
        }

        // Mark result storage as being on the device
        gpuResultStorage.DataLocation = DataLocation.Device;
    }

    private void AddToResult(IVolumeStorage<T> result, T beta)
    {
        if (result is not GpuStorage<T> gpuResult)
        {
            throw new ArgumentException("Result must be of type GpuStorage<T>.");
        }

        // Ensure tensors are on the device
        this.CopyToDevice();
        gpuResult.CopyToDevice();

        // Compute broadcasted shape
        var broadcastedShape = Shape.GetBroadcastedShape(this.Shape, gpuResult.Shape);
        if (!gpuResult.Shape.Equals(broadcastedShape))
        {
            Console.WriteLine("Broadcast Validation Failed:");
            Console.WriteLine($"  Source Shape: {this.Shape}");
            Console.WriteLine($"  Result Shape: {gpuResult.Shape}");
            Console.WriteLine($"  Broadcasted Shape: {broadcastedShape}");
            throw new ArgumentException(
                $"Result shape {gpuResult.Shape} does not match the broadcasted shape {broadcastedShape}.");
        }

        // Compute strides for the broadcasted shape
        var broadcastedStrides = broadcastedShape.GetStrides();
        var sourceStrides = this.Shape.GetStrides();
        var adjustedStrides = new int[4];

        for (var i = 0; i < adjustedStrides.Length; i++)
        {
            adjustedStrides[i] = this.Shape.Dimensions[i] == 1 ? 0 : sourceStrides[i];
        }

        // Debugging output
        Console.WriteLine($"Adjusted Strides for Source Tensor: {string.Join(", ", adjustedStrides)}");
        Console.WriteLine($"Broadcasted Strides: {string.Join(", ", broadcastedStrides)}");

        // Configure tensor descriptors
        using var thisDesc = new TensorDescriptor();
        using var resultDesc = new TensorDescriptor();

        // Set source tensor descriptor
        thisDesc.SetTensor4dDescriptorEx(
            GetCudnnDataType(),
            broadcastedShape.Dimensions[3], broadcastedShape.Dimensions[2],
            broadcastedShape.Dimensions[1], broadcastedShape.Dimensions[0],
            adjustedStrides[3], adjustedStrides[2],
            adjustedStrides[1], adjustedStrides[0]);

        // Set result tensor descriptor
        resultDesc.SetTensor4dDescriptorEx(
            GetCudnnDataType(),
            gpuResult.Shape.Dimensions[3], gpuResult.Shape.Dimensions[2],
            gpuResult.Shape.Dimensions[1], gpuResult.Shape.Dimensions[0],
            broadcastedStrides[3], broadcastedStrides[2],
            broadcastedStrides[1], broadcastedStrides[0]);

        // Perform addition
        if (typeof(T) == typeof(float))
        {
            var alpha = Convert.ToSingle(T.One);
            var betaConverted = Convert.ToSingle(beta);

            GpuContext.Instance.CudnnContext.AddTensor(
                alpha,
                thisDesc,
                this._deviceData as CudaDeviceVariable<float>,
                betaConverted,
                resultDesc,
                gpuResult._deviceData as CudaDeviceVariable<float>);
        }
        else if (typeof(T) == typeof(double))
        {
            var alpha = Convert.ToDouble(T.One);
            var betaConverted = Convert.ToDouble(beta);

            GpuContext.Instance.CudnnContext.AddTensor(
                alpha,
                thisDesc,
                this._deviceData as CudaDeviceVariable<double>,
                betaConverted,
                resultDesc,
                gpuResult._deviceData as CudaDeviceVariable<double>);
        }
        else
        {
            throw new NotSupportedException($"Unsupported type {typeof(T)}");
        }

        // Mark result as being on the device
        gpuResult.DataLocation = DataLocation.Device;
    }


    private static cudnnDataType GetCudnnDataType()
    {
        if (typeof(T) == typeof(float))
        {
            return cudnnDataType.Float;
        }

        if (typeof(T) == typeof(double))
        {
            return cudnnDataType.Double;
        }

        throw new NotSupportedException($"Data type {typeof(T)} is not supported.");
    }

    /// <summary>
    ///     Ensures the data is present on the GPU device by copying it from the host if necessary.
    /// </summary>
    public void CopyToDevice()
    {
        if (this.DataLocation == DataLocation.Device)
        {
            return; // Data is already on the device
        }

        this._deviceData.CopyToDevice(this._hostData);
        this.DataLocation = DataLocation.Device;
    }

    /// <summary>
    ///     Copies the data from another storage to this storage.
    /// </summary>
    /// <param name="source">The source storage to copy from.</param>
    public void CopyFrom(IVolumeStorage<T> source)
    {
        this.EnsureNotDisposed();

        if (source is not GpuStorage<T> gpuSource)
        {
            throw new ArgumentException("Source must be of type GpuStorage<T>.", nameof(source));
        }

        // Ensure both source and destination data are on the device
        gpuSource.CopyToDevice();
        this.CopyToDevice();

        // Validate that the source and destination sizes match
        if (this._deviceData.Size != gpuSource._deviceData.Size)
        {
            throw new InvalidOperationException("Source and destination sizes must match.");
        }

        // Perform the device-to-device copy
        var result = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(
            this._deviceData.DevicePointer,
            gpuSource._deviceData.DevicePointer,
            new SizeT(this._deviceData.SizeInBytes),
            GpuContext.Instance.DefaultStream.Stream
        );

        if (result != CUResult.Success)
        {
            throw new CudaException($"Device-to-device memory copy failed with error code: {result}");
        }

        // Mark data as being on the device
        this.DataLocation = DataLocation.Device;
    }

    /// <summary>
    ///     Ensures that the object is not disposed before performing an operation.
    /// </summary>
    private void EnsureNotDisposed()
    {
        if (this._isDisposed)
        {
            throw new ObjectDisposedException(nameof(GpuStorage<T>));
        }
    }

    private void EnsureHostData()
    {
        if (this.DataLocation == DataLocation.Device)
        {
            this._deviceData.CopyToHost(this._hostData);
            this.DataLocation = DataLocation.Host;
        }
    }
}