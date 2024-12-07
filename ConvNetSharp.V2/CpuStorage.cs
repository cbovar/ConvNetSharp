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
    public IVolumeStorage<T> MultiplyByScalar(T scalar, bool inPlace = false)
    {
        // If not in-place, create a copy of the storage
        var resultStorage = inPlace ? this : new CpuStorage<T>(this.GetData().ToArray(), this.Shape);

        // Apply scaling operation
        for (var i = 0; i < resultStorage.GetData().Length; i++)
        {
            resultStorage.GetData()[i] *= scalar;
        }

        return resultStorage;
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
}