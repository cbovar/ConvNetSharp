using System.Numerics;

namespace ConvNetSharp.V2;

/// <summary>
///     Represents a CPU-based storage for volumes in ConvNetSharp.
///     This class manages data storage and operations on the CPU.
/// </summary>
/// <typeparam name="T">The data type of the storage, must be a struct and support numerical operations.</typeparam>
public class CpuStorage<T> : IVolumeStorage<T>
    where T : struct, INumber<T>
{
    private readonly T[] _data; // Private field to store the data

    /// <summary>
    ///     Initializes a new instance of the <see cref="CpuStorage{T}" /> class.
    /// </summary>
    /// <param name="data">The initial data for the storage.</param>
    /// <param name="shape">The shape of the volume.</param>
    public CpuStorage(T[] data, Shape shape)
    {
        if (data.Length != shape.TotalLength)
        {
            throw new ArgumentException("Data length does not match the total length of the shape.");
        }

        this._data = data;
        this.Shape = shape;
    }

    /// <summary>
    ///     Initializes a new instance of the <see cref="CpuStorage{T}" /> class with a specified shape.
    /// </summary>
    /// <param name="shape">The shape of the tensor represented by this storage.</param>
    public CpuStorage(Shape shape)
        : this(null, shape) // Delegate to the other constructor
    {
    }

    /// <summary>
    ///     Gets the device type where the data is stored (e.g., CPU or GPU).
    /// </summary>
    public Device Device => Device.CPU;

    /// <summary>
    ///     Gets the shape of the volume.
    /// </summary>
    public Shape Shape { get; }

    /// <summary>
    ///     Returns the data array stored in this CPU storage.
    /// </summary>
    /// <returns>The data array.</returns>
    public T[] GetData()
    {
        return this._data;
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
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data), "Data cannot be null.");
        }

        if (data.Length != this.Shape.TotalLength)
        {
            throw new ArgumentException(
                $"The length of the provided data ({data.Length}) does not match the total length of the volume ({this.Shape.TotalLength}).");
        }

        Array.Copy(data, this._data, data.Length);
    }

    /// <summary>
    ///     Applies a user-defined operation to the data.
    /// </summary>
    /// <param name="operation">The operation to apply to each element of the storage.</param>
    public void ApplyOperation(Func<T, T> operation)
    {
        for (var i = 0; i < this._data.Length; i++)
        {
            this._data[i] = operation(this._data[i]);
        }
    }

    /// <summary>
    ///     Adds the data from another storage to the current storage and places the result in a third storage.
    /// </summary>
    /// <param name="other">The other storage to add.</param>
    /// <param name="result">The storage where the result is stored.</param>
    /// <exception cref="ArgumentException">Thrown if the storage types or shapes do not match.</exception>
    public void Add(IVolumeStorage<T> other, IVolumeStorage<T> result)
    {
        if (other is not CpuStorage<T> cpuOther || result is not CpuStorage<T> cpuResult)
        {
            throw new ArgumentException("Both other and result must be of type CpuStorage<T>.");
        }

        var inputShape = this.Shape;
        var otherShape = cpuOther.Shape;
        var resultShape = cpuResult.Shape;

        if (!inputShape.IsBroadcastableTo(otherShape) ||
            !resultShape.Equals(Shape.GetBroadcastedShape(inputShape, otherShape)))
        {
            throw new ArgumentException("Shapes of storages are not broadcastable.");
        }

        var inputData = this._data;
        var otherData = cpuOther._data;
        var resultData = cpuResult._data;

        var inputStrides = inputShape.GetStrides();
        var otherStrides = otherShape.GetStrides();
        var resultStrides = resultShape.GetStrides();

        var inputIndex = new int[inputShape.Dimensions.Length];
        var otherIndex = new int[otherShape.Dimensions.Length];

        for (var i = 0; i < resultShape.TotalLength; i++)
        {
            var resultCoords = resultShape.GetCoordinates(i, resultStrides);
            inputIndex = inputShape.GetBroadcastCoordinates(resultCoords, inputStrides);
            otherIndex = otherShape.GetBroadcastCoordinates(resultCoords, otherStrides);

            resultData[i] = inputData[inputShape.GetFlatIndex(inputIndex)] +
                            otherData[otherShape.GetFlatIndex(otherIndex)];
        }
    }

    /// <summary>
    ///     Clears the storage data by setting all values to zero.
    /// </summary>
    public void Clear()
    {
        for (var i = 0; i < this._data.Length; i++)
        {
            this._data[i] = T.Zero;
        }
    }

    public void Convolution(IVolumeStorage<T> filterStorage, int xpad, int ypad, int stride,
        IVolumeStorage<T> outputStorage)
    {
        if (filterStorage is not CpuStorage<T> filterCpuStorage || outputStorage is not CpuStorage<T> outputCpuStorage)
        {
            throw new ArgumentException("Both filter and output storages must be of type CpuStorage<T>.");
        }

        var inputShape = this.Shape;
        var filterShape = filterCpuStorage.Shape;

        if (inputShape.Dimensions[2] != filterShape.Dimensions[2])
        {
            throw new ArgumentException("Input depth and filter depth must match.");
        }

        var inputWidth = inputShape.Dimensions[0];
        var inputHeight = inputShape.Dimensions[1];
        var inputDepth = inputShape.Dimensions[2];
        var batchSize = inputShape.Dimensions[3];

        var filterWidth = filterShape.Dimensions[0];
        var filterHeight = filterShape.Dimensions[1];
        var filterDepth = filterShape.Dimensions[2];
        var outputDepth = filterShape.Dimensions[3];

        var outputWidth = outputCpuStorage.Shape.Dimensions[0];
        var outputHeight = outputCpuStorage.Shape.Dimensions[1];

        for (var n = 0; n < batchSize; n++)
        {
            for (var depth = 0; depth < outputDepth; depth++)
            {
                var y = -ypad;
                for (var ay = 0; ay < outputHeight; y += stride, ay++)
                {
                    var x = -xpad;
                    for (var ax = 0; ax < outputWidth; x += stride, ax++)
                    {
                        var sum = T.Zero;
                        for (var fy = 0; fy < filterHeight; fy++)
                        {
                            var oy = y + fy;
                            for (var fx = 0; fx < filterWidth; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                {
                                    for (var fd = 0; fd < filterDepth; fd++)
                                    {
                                        sum += filterCpuStorage.Get(fx, fy, fd, depth) *
                                               this.Get(ox, oy, fd, n);
                                    }
                                }
                            }
                        }

                        outputCpuStorage.Set(ax, ay, depth, n, sum);
                    }
                }
            }
        }
    }

    /// <summary>
    ///     Repeats the elements of this volume according to the specified replication factors (reps),
    ///     and stores the result in the provided result volume.
    /// </summary>
    /// <param name="reps">The replication factors volume, which determines how many times to replicate the data.</param>
    /// <param name="result">The volume to store the tiled output. Its shape must match the expected tiled dimensions.</param>
    /// <exception cref="ArgumentException">
    ///     Thrown if the dimensions of the input and result do not match expected tiling
    ///     rules.
    /// </exception>
    public void Tile(IVolumeStorage<T> reps, IVolumeStorage<T> result)
    {
        // Original dimensions of the input volume
        var batchSize = this.Shape.Dimensions[3];
        var channel = this.Shape.Dimensions[2];
        var height = this.Shape.Dimensions[1];
        var width = this.Shape.Dimensions[0];

        // Dimensions of the output result volume
        var outputBatchSize = result.Shape.Dimensions[3];
        var outputChannel = result.Shape.Dimensions[2];
        var outputHeight = result.Shape.Dimensions[1];
        var outputWidth = result.Shape.Dimensions[0];

        // Iterate through each element of the result volume
        for (var n = 0; n < outputBatchSize; n++) // Iterate over batch size
        {
            for (var c = 0; c < outputChannel; c++) // Iterate over channels
            {
                for (var h = 0; h < outputHeight; h++) // Iterate over height
                {
                    for (var w = 0; w < outputWidth; w++) // Iterate over width
                    {
                        // Calculate the corresponding position in the original volume using modulo
                        // This ensures wrapping behavior for tiling
                        result.Set(w, h, c, n, this.Get(w % width, h % height, c % channel, n % batchSize));
                    }
                }
            }
        }
    }

    /// <summary>
    ///     Multiplies all elements of this volume by a scalar value and stores the result in the provided CPU storage.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <param name="resultStorage">The CPU storage where the result of the operation will be stored.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="resultStorage" /> is null.</exception>
    /// <exception cref="ArgumentException">
    ///     Thrown when <paramref name="resultStorage" /> is not of type <see cref="CpuStorage{T}" /> or when the shapes do not
    ///     match.
    /// </exception>
    public void MultiplyByScalar(T scalar, IVolumeStorage<T> resultStorage)
    {
        if (resultStorage == null)
        {
            throw new ArgumentNullException(nameof(resultStorage), "Result storage cannot be null.");
        }

        if (resultStorage is not CpuStorage<T> cpuResultStorage)
        {
            throw new ArgumentException("Result storage must be of type CpuStorage<T>.", nameof(resultStorage));
        }

        // Ensure the shapes match
        if (!this.Shape.Equals(cpuResultStorage.Shape))
        {
            throw new ArgumentException(
                $"Shapes do not match. Source: {this.Shape}, Result: {cpuResultStorage.Shape}.");
        }

        var resultData = cpuResultStorage.GetData();

        // Copy data if resultStorage is not the same as this
        if (!ReferenceEquals(this, cpuResultStorage))
        {
            Array.Copy(this.GetData(), resultData, this.GetData().Length);
        }

        // Apply the scaling operation
        for (var i = 0; i < resultData.Length; i++)
        {
            resultData[i] = resultData[i] * scalar;
        }
    }

    /// <summary>
    ///     Subtracts the elements of another volume from this volume, storing the result in a third volume.
    /// </summary>
    /// <param name="rightStorage">The volume to subtract from this volume.</param>
    /// <param name="resultStorage">The volume to store the result.</param>
    public void Subtract(IVolumeStorage<T> rightStorage, IVolumeStorage<T> resultStorage)
    {
        if (rightStorage is not CpuStorage<T> rightCpu || resultStorage is not CpuStorage<T> resultCpu)
        {
            throw new ArgumentException("Both rightStorage and resultStorage must be of type CpuStorage<T>.");
        }

        var leftData = this.GetData();
        var rightData = rightCpu.GetData();
        var resultData = resultCpu.GetData();

        if (leftData.Length != rightData.Length || leftData.Length != resultData.Length)
        {
            throw new ArgumentException("All storages must have the same size.");
        }

        for (var i = 0; i < leftData.Length; i++)
        {
            resultData[i] = leftData[i] - rightData[i];
        }
    }

    public void MatMultiply(IVolumeStorage<T> right, IVolumeStorage<T> result)
    {
        if (right is not CpuStorage<T> cpuRight || result is not CpuStorage<T> cpuResult)
        {
            throw new ArgumentException("Both right and result must be of type CpuStorage<T>.");
        }

        // Validate shapes for matrix multiplication
        if (this.Shape.Dimensions[0] != cpuRight.Shape.Dimensions[1])
        {
            throw new ArgumentException(
                $"Matrix dimensions do not align for multiplication. Left: {this.Shape}, Right: {cpuRight.Shape}");
        }

        var batchSize = Math.Max(this.Shape.Dimensions[3], cpuRight.Shape.Dimensions[3]);
        if (batchSize != result.Shape.Dimensions[3])
        {
            throw new ArgumentException("Batch size mismatch between operands and result.");
        }

        var resultShape = ComputeMatMultiplyShape(this.Shape, cpuRight.Shape);
        if (!result.Shape.Equals(resultShape))
        {
            throw new ArgumentException($"Result shape should be {resultShape}, but is {result.Shape}");
        }

        // Perform matrix multiplication
        var m = this.Shape.Dimensions[1]; // Rows in left matrix
        var n = cpuRight.Shape.Dimensions[0]; // Columns in right matrix
        var k = this.Shape.Dimensions[0]; // Shared dimension

        for (var b = 0; b < batchSize; b++)
        {
            var leftBatch = this.Shape.Dimensions[3] == 1 ? 0 : b;
            var rightBatch = cpuRight.Shape.Dimensions[3] == 1 ? 0 : b;

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var sum = T.Zero;

                    for (var shared = 0; shared < k; shared++)
                    {
                        var leftValue = this.Get(shared, row, 0, leftBatch);
                        var rightValue = cpuRight.Get(col, shared, 0, rightBatch);
                        sum += leftValue * rightValue;
                    }

                    result.Set(col, row, 0, b, sum);
                }
            }
        }
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
        if (rightStorage == null)
        {
            throw new ArgumentNullException(nameof(rightStorage));
        }

        if (resultStorage == null)
        {
            throw new ArgumentNullException(nameof(resultStorage));
        }

        // Validate that shapes are broadcast-compatible
        if (!this.Shape.IsBroadcastableTo(rightStorage.Shape) || !this.Shape.IsBroadcastableTo(resultStorage.Shape))
        {
            throw new ArgumentException("Shapes are incompatible for broadcasting.");
        }

        var leftData = this.GetData();
        var rightData = rightStorage.GetData();
        var resultData = resultStorage.GetData();

        // Perform element-wise multiplication with broadcasting
        for (var i = 0; i < resultData.Length; i++)
        {
            var leftIndex = this.Shape.GetBroadcastIndex(i, resultStorage.Shape);
            var rightIndex = rightStorage.Shape.GetBroadcastIndex(i, resultStorage.Shape);

            resultData[i] = leftData[leftIndex] * rightData[rightIndex];
        }

        // Write back the result to the storage
        resultStorage.SetData(resultData);
    }

    /// <summary>
    ///     Computes the transpose of the current volume storage and stores it in the provided result storage.
    /// </summary>
    /// <param name="resultStorage">The volume storage where the transposed data will be stored.</param>
    /// <exception cref="ArgumentNullException">
    ///     Thrown when <paramref name="resultStorage" /> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    ///     Thrown when the shapes of the volume storages are not compatible for transposition.
    /// </exception>
    /// <remarks>
    ///     The method computes the transpose by swapping the first two dimensions of the shape.
    ///     The operation requires the shapes of the current storage and result storage to be compatible.
    /// </remarks>
    public void Transpose(IVolumeStorage<T> resultStorage)
    {
        if (resultStorage == null)
        {
            throw new ArgumentNullException(nameof(resultStorage));
        }

        // Validate that shapes are compatible for transposition
        var expectedShape = new Shape(this.Shape.Dimensions[1], this.Shape.Dimensions[0], this.Shape.Dimensions[2],
            this.Shape.Dimensions[3]);
        if (!resultStorage.Shape.Equals(expectedShape))
        {
            throw new ArgumentException(
                $"Shapes are not compatible for transposition. Expected shape: {expectedShape}, but got: {resultStorage.Shape}.");
        }

        var inputData = this.GetData();
        var resultData = resultStorage.GetData();

        // Perform transposition
        for (var n = 0; n < this.Shape.Dimensions[3]; n++)
        {
            for (var c = 0; c < this.Shape.Dimensions[2]; c++)
            {
                for (var h = 0; h < this.Shape.Dimensions[1]; h++)
                {
                    for (var w = 0; w < this.Shape.Dimensions[0]; w++)
                    {
                        var inputIndex = n * this.Shape.Dimensions[2] * this.Shape.Dimensions[1] *
                                         this.Shape.Dimensions[0]
                                         + c * this.Shape.Dimensions[1] * this.Shape.Dimensions[0]
                                         + h * this.Shape.Dimensions[0]
                                         + w;

                        var resultIndex = n * this.Shape.Dimensions[2] * this.Shape.Dimensions[0] *
                                          this.Shape.Dimensions[1]
                                          + c * this.Shape.Dimensions[0] * this.Shape.Dimensions[1]
                                          + w * this.Shape.Dimensions[1]
                                          + h;

                        resultData[resultIndex] = inputData[inputIndex];
                    }
                }
            }
        }

        // Write back the result to the storage
        resultStorage.SetData(resultData);
    }

    /// <summary>
    ///     Retrieves a single value from the storage by its flat index.
    /// </summary>
    public T Get(int i)
    {
        if (i < 0 || i >= this._data.Length)
        {
            throw new IndexOutOfRangeException("Index out of bounds.");
        }

        return this._data[i];
    }

    /// <summary>
    ///     Retrieves a single value from the storage based on 2D coordinates (width and height).
    /// </summary>
    public T Get(int w, int h)
    {
        var index = h * this.Shape.Dimensions[0] + w;
        return this.Get(index);
    }

    /// <summary>
    ///     Retrieves a single value from the storage based on 3D coordinates (width, height, and channel).
    /// </summary>
    public T Get(int w, int h, int c)
    {
        var index = c * this.Shape.Dimensions[0] * this.Shape.Dimensions[1]
                    + h * this.Shape.Dimensions[0]
                    + w;
        return this.Get(index);
    }

    /// <summary>
    ///     Retrieves a single value from the storage based on 4D coordinates (width, height, channel, and batch).
    /// </summary>
    public T Get(int w, int h, int c, int n)
    {
        if (w < 0 || h < 0 || c < 0 || n < 0 ||
            w >= this.Shape.Dimensions[0] ||
            h >= this.Shape.Dimensions[1] ||
            c >= this.Shape.Dimensions[2] ||
            n >= this.Shape.Dimensions[3])
        {
            throw new IndexOutOfRangeException(
                $"Index out of bounds for shape {this.Shape}: (w={w}, h={h}, c={c}, n={n})");
        }

        var index = n * this.Shape.Dimensions[0] * this.Shape.Dimensions[1] * this.Shape.Dimensions[2]
                    + c * this.Shape.Dimensions[0] * this.Shape.Dimensions[1]
                    + h * this.Shape.Dimensions[0]
                    + w;
        return this.Get(index);
    }

    /// <summary>
    ///     Retrieves a single value from the storage based on an n-dimensional coordinate array.
    /// </summary>
    public T Get(int[] coordinates)
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

        return this.Get(index);
    }

    /// <summary>
    ///     Sets a single value in the storage at the specified flat index.
    /// </summary>
    public void Set(int index, T value)
    {
        if (index < 0 || index >= this._data.Length)
        {
            throw new IndexOutOfRangeException("Index out of bounds.");
        }

        this._data[index] = value;
    }

    /// <summary>
    ///     Sets a single value in the storage based on 2D coordinates (width and height).
    /// </summary>
    public void Set(int w, int h, T value)
    {
        var index = h * this.Shape.Dimensions[0] + w;
        this.Set(index, value);
    }

    /// <summary>
    ///     Sets a single value in the storage based on 3D coordinates (width, height, and channel).
    /// </summary>
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
    public void Set(int w, int h, int c, int n, T value)
    {
        if (w < 0 || h < 0 || c < 0 || n < 0 ||
            w >= this.Shape.Dimensions[0] ||
            h >= this.Shape.Dimensions[1] ||
            c >= this.Shape.Dimensions[2] ||
            n >= this.Shape.Dimensions[3])
        {
            throw new IndexOutOfRangeException(
                $"Index out of bounds for shape {this.Shape}: (w={w}, h={h}, c={c}, n={n})");
        }

        var index = n * this.Shape.Dimensions[0] * this.Shape.Dimensions[1] * this.Shape.Dimensions[2]
                    + c * this.Shape.Dimensions[0] * this.Shape.Dimensions[1]
                    + h * this.Shape.Dimensions[0]
                    + w;
        this.Set(index, value);
    }

    /// <summary>
    ///     Sets a single value in the storage based on an n-dimensional coordinate array.
    /// </summary>
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
    ///     Accumulates the data from the current storage into another storage in-place.
    /// </summary>
    /// <param name="result">The storage where the data is accumulated.</param>
    /// <exception cref="ArgumentException">Thrown if the storage types or shapes do not match.</exception>
    public void Add(IVolumeStorage<T> result)
    {
        if (result is not CpuStorage<T> cpuResult)
        {
            throw new ArgumentException("Result must be of type CpuStorage<T>.");
        }

        if (this._data.Length != cpuResult._data.Length)
        {
            throw new ArgumentException("Shapes of storages must match.");
        }

        var resultData = cpuResult._data;

        // In-place accumulation: result += this
        for (var i = 0; i < this._data.Length; i++)
        {
            resultData[i] += this._data[i];
        }
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
        if (power == null)
        {
            throw new ArgumentNullException(nameof(power), "Exponent storage cannot be null.");
        }

        if (result == null)
        {
            throw new ArgumentNullException(nameof(result), "Result storage cannot be null.");
        }

        if (power is not CpuStorage<T> exponentStorage || result is not CpuStorage<T> resultCpuStorage)
        {
            throw new ArgumentException("Both power and result must be of type CpuStorage<T>.");
        }

        if (!this.Shape.IsBroadcastableTo(power.Shape) || !this.Shape.IsBroadcastableTo(result.Shape))
        {
            throw new ArgumentException("Shapes are not compatible for broadcasting.");
        }

        var baseData = this.GetData();
        var exponentData = exponentStorage.GetData();
        var resultData = resultCpuStorage.GetData();

        for (var i = 0; i < resultData.Length; i++)
        {
            var baseIndex = this.Shape.GetBroadcastIndex(i, resultCpuStorage.Shape);
            var exponentIndex = power.Shape.GetBroadcastIndex(i, resultCpuStorage.Shape);

            // Convert to double, perform Math.Pow, and convert back to T
            var baseValue = Convert.ToDouble(baseData[baseIndex]);
            var exponentValue = Convert.ToDouble(exponentData[exponentIndex]);
            resultData[i] = T.CreateChecked(Math.Pow(baseValue, exponentValue));
        }

        result.SetData(resultData);
    }

    /// <summary>
    ///     Computes the expected shape of the result of a matrix multiplication.
    /// </summary>
    /// <param name="leftShape">The shape of the left matrix.</param>
    /// <param name="rightShape">The shape of the right matrix.</param>
    /// <returns>The expected shape of the result.</returns>
    public static Shape ComputeMatMultiplyShape(Shape leftShape, Shape rightShape)
    {
        if (leftShape.Dimensions[0] != rightShape.Dimensions[1])
        {
            throw new ArgumentException("Matrix dimensions do not align for multiplication.");
        }

        var batchSize = Math.Max(leftShape.Dimensions[3], rightShape.Dimensions[3]);
        return new Shape(
            rightShape.Dimensions[0], // Columns of the result
            leftShape.Dimensions[1], // Rows of the result
            1, // Depth remains 1
            batchSize // Batch size
        );
    }
}