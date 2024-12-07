using System.Numerics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaDNN;

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

        if (!this.Shape.Equals(gpuOther.Shape) || !this.Shape.Equals(gpuResult.Shape))
        {
            throw new ArgumentException("Shapes of all storages must match.");
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

    public void Tile(IVolumeStorage<T> reps, IVolumeStorage<T> result)
    {
        throw new NotImplementedException();
    }

    public IVolumeStorage<T> MultiplyByScalar(T scalar, bool inPlace = false)
    {
        // Ensure the data is on the device
        this.CopyToDevice();

        // If not in-place, create result storage and copy the current data
        var resultStorage = inPlace ? this : new GpuStorage<T>(this.Shape);
        if (!inPlace)
        {
            resultStorage.CopyFrom(this);
        }

        // Perform scaling based on the type of T
        if (typeof(T) == typeof(float))
        {
            GpuContext.Instance.CublasContext.Scale(
                Convert.ToSingle(scalar),
                resultStorage._deviceData as CudaDeviceVariable<float>,
                1);
        }
        else if (typeof(T) == typeof(double))
        {
            GpuContext.Instance.CublasContext.Scale(
                Convert.ToDouble(scalar),
                resultStorage._deviceData as CudaDeviceVariable<double>,
                1);
        }
        else
        {
            throw new NotSupportedException($"Type {typeof(T)} is not supported for GPU operations.");
        }

        // Return the result storage (new or in-place)
        return resultStorage;
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

    /// <summary>
    ///     A helper method to accumulate this storage into a result tensor.
    ///     Ensures that both this storage and the result are synchronized to the GPU.
    ///     Uses cuDNN's AddTensor method for GPU-accelerated addition.
    /// </summary>
    /// <param name="result">The result tensor storage.</param>
    /// <param name="beta">
    ///     The scale factor for the current data in the result tensor. Default is 1.0,
    ///     meaning the existing data in the result is retained and accumulated.
    /// </param>
    /// <exception cref="ArgumentException">Thrown if result is not of type GpuStorage or shapes are incompatible.</exception>
    private void AddToResult(IVolumeStorage<T> result, T beta)
    {
        if (result is not GpuStorage<T> gpuResult)
        {
            throw new ArgumentException("Result must be of type GpuStorage<T>.");
        }

        // Ensure both tensors are on the device
        this.CopyToDevice();
        gpuResult.CopyToDevice();

        // Validate shape compatibility
        if (!this.Shape.Equals(gpuResult.Shape))
        {
            throw new ArgumentException("Shapes of the input and result tensors must match.");
        }

        // Initialize tensor descriptors
        using var thisDesc = new TensorDescriptor();
        using var resultDesc = new TensorDescriptor();

        var dimensions = this.Shape.Dimensions;
        thisDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, GetCudnnDataType(),
            dimensions[3], dimensions[2], dimensions[1], dimensions[0]);
        resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, GetCudnnDataType(),
            dimensions[3], dimensions[2], dimensions[1], dimensions[0]);

        // Convert T to a specific type
        if (typeof(T) == typeof(float))
        {
            var alpha = Convert.ToSingle(T.One); // Scale for this tensor
            var betaConverted = Convert.ToSingle(beta); // Scale for the result tensor
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
            var alpha = Convert.ToDouble(T.One); // Scale for this tensor
            var betaConverted = Convert.ToDouble(beta); // Scale for the result tensor
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

        // Mark result data as being on the device
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