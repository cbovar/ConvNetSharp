using System.Numerics;

namespace ConvNetSharp.V2;

/// <summary>
///     Represents a multidimensional volume of data, supporting operations such as creation,
///     manipulation, and mathematical computations. This class abstracts the storage mechanism
///     (CPU or GPU) and provides a unified interface for handling data in neural networks.
/// </summary>
/// <typeparam name="T">The numeric type of the data stored in the volume (e.g., float, double).</typeparam>
public class Volume<T> : IDisposable where T : struct, INumber<T>
{
    private readonly IVolumeStorage<T> _storage;
    private bool _isDisposed;

    /// <summary>
    ///     Initializes a new instance of the <see cref="Volume{T}" /> class.
    /// </summary>
    /// <param name="storage">The underlying storage implementation for the volume.</param>
    /// <param name="shape">The shape of the volume.</param>
    private Volume(IVolumeStorage<T> storage, Shape shape)
    {
        this._storage = storage ?? throw new ArgumentNullException(nameof(storage));
        this.Shape = shape;
    }

    /// <summary>
    ///     Gets the shape of the volume.
    /// </summary>
    public Shape Shape { get; }

    /// <summary>
    ///     Releases the resources used by the volume.
    /// </summary>
    public void Dispose()
    {
        this.Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    ///     Factory method to create a volume from existing data.
    /// </summary>
    /// <param name="data">The data to populate the volume with.</param>
    /// <param name="shape">The shape of the volume.</param>
    /// <param name="device">The device to use for storage (CPU or GPU).</param>
    /// <returns>A new volume instance populated with the provided data.</returns>
    /// <exception cref="ArgumentException">Thrown if the device type is unsupported.</exception>
    public static Volume<T> Create(T[] data, Shape shape, Device device = Device.CPU)
    {
        IVolumeStorage<T> storage = device switch
        {
            Device.CPU => new CpuStorage<T>(data, shape),
            Device.GPU => new GpuStorage<T>(data, shape),
            _ => throw new ArgumentException("Unsupported device type")
        };

        return new Volume<T>(storage, shape);
    }

    /// <summary>
    ///     Factory method to create an empty volume with uninitialized data.
    /// </summary>
    /// <param name="shape">The shape of the volume.</param>
    /// <param name="device">The device to use for storage (CPU or GPU).</param>
    /// <returns>A new volume instance with uninitialized data.</returns>
    public static Volume<T> CreateEmpty(Shape shape, Device device = Device.CPU)
    {
        var data = new T[shape.TotalLength];
        return Create(data, shape, device);
    }

    /// <summary>
    ///     Retrieves the data stored in the volume.
    /// </summary>
    /// <returns>An array containing the volume's data.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the volume is disposed.</exception>
    public T[] GetData()
    {
        this.ThrowIfDisposed();
        return this._storage.GetData();
    }

    /// <summary>
    ///     Applies a specified operation to each element in the volume.
    /// </summary>
    /// <param name="operation">The operation to apply.</param>
    /// <exception cref="ObjectDisposedException">Thrown if the volume is disposed.</exception>
    public void ApplyOperation(Func<T, T> operation)
    {
        this.ThrowIfDisposed();
        this._storage.ApplyOperation(operation);
    }

    /// <summary>
    ///     Adds the elements of another volume to this volume, storing the result in a third volume.
    /// </summary>
    /// <param name="other">The volume to add.</param>
    /// <param name="result">The volume to store the result.</param>
    /// <exception cref="ObjectDisposedException">Thrown if any of the volumes are disposed.</exception>
    public void Add(Volume<T> other, Volume<T> result)
    {
        this.ThrowIfDisposed();
        other.ThrowIfDisposed();
        result.ThrowIfDisposed();

        this._storage.Add(other._storage, result._storage);
    }

    /// <summary>
    ///     Clears the volume by setting all elements to zero.
    /// </summary>
    /// <exception cref="ObjectDisposedException">Thrown if the volume is disposed.</exception>
    public void Clear()
    {
        this.ThrowIfDisposed();
        this._storage.Clear();
    }

    /// <summary>
    ///     Disposes of the volume's resources.
    /// </summary>
    /// <param name="disposing">Indicates whether managed resources should also be disposed.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!this._isDisposed && disposing)
        {
            if (this._storage is IDisposable disposableStorage)
            {
                disposableStorage.Dispose();
            }

            this._isDisposed = true;
        }
    }

    /// <summary>
    ///     Throws an exception if the volume has been disposed.
    /// </summary>
    private void ThrowIfDisposed()
    {
        if (this._isDisposed)
        {
            throw new ObjectDisposedException(nameof(Volume<T>));
        }
    }

    /// <summary>
    ///     Finalizer to ensure resources are released if Dispose is not called.
    /// </summary>
    ~Volume()
    {
        this.Dispose(false);
    }

    public void Tile(Volume<T> reps, Volume<T> result)
    {
        this.ThrowIfDisposed();
        this._storage.Tile(reps._storage, result._storage);
    }

    public void Log(Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void ConvolutionGradient(Volume<T> filter, Volume<T> outputGradients, Volume<T> filterGradient, int pad,
        int stride, Volume<T> inputGradient)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///     Performs a convolution operation with the given parameters and stores the result.
    /// </summary>
    /// <param name="evaluate">The volume to convolve with.</param>
    /// <param name="xpad">Padding in the horizontal direction.</param>
    /// <param name="ypad">Padding in the vertical direction.</param>
    /// <param name="stride">Stride for the convolution.</param>
    /// <param name="result">The volume to store the convolution result.</param>
    /// <exception cref="ObjectDisposedException">Thrown if any of the volumes are disposed.</exception>
    public void Convolution(Volume<T> evaluate, int xpad, int ypad, int stride, Volume<T> result)
    {
        this.ThrowIfDisposed();
        evaluate.ThrowIfDisposed();
        result.ThrowIfDisposed();

        this._storage.Convolution(evaluate._storage, xpad, ypad, stride, result._storage);
    }

    public void Concat(Volume<T> right, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///     Reshapes the volume to a new shape, preserving the data order.
    /// </summary>
    /// <param name="dimensions">The new dimensions for the volume.</param>
    /// <returns>A new volume with the specified shape.</returns>
    public Volume<T> ReShape(params int[] dimensions)
    {
        return this.ReShape(Shape.From(dimensions));
    }

    /// <summary>
    ///     Reshapes the volume to a new shape, preserving the data order.
    /// </summary>
    /// <param name="shape">The new shape for the volume.</param>
    /// <returns>A new volume with the specified shape.</returns>
    public Volume<T> ReShape(Shape shape)
    {
        var guessedShape = new Shape(shape);

        for (var i = 0; i < 4; i++)
        {
            if (shape.Dimensions[i] == Shape.Keep)
            {
                guessedShape.SetDimension(i, this.Shape.Dimensions[i]);
            }
        }

        guessedShape.GuessUnknownDimension(this.Shape.TotalLength);

        return new Volume<T>(this._storage, guessedShape);
    }

    /// <summary>
    ///     Factory method to create a volume with random data.
    /// </summary>
    public static Volume<T> CreateRandom(Shape shape, double mean, double scale, Device device = Device.CPU)
    {
        var random = new Random();
        var data = new T[shape.TotalLength];

        for (var i = 0; i < data.Length; i++)
        {
            // Generate random data with a normal distribution
            var gaussian = mean + scale * Math.Sqrt(-2.0 * Math.Log(random.NextDouble())) *
                Math.Sin(2.0 * Math.PI * random.NextDouble());
            data[i] = T.CreateChecked(gaussian);
        }

        return Create(data, shape, device);
    }

    /// <summary>
    ///     Creates a volume with all elements initialized to one.
    /// </summary>
    /// <param name="shape">The shape of the volume.</param>
    /// <returns>A volume filled with ones.</returns>
    public static Volume<T> Ones(Shape shape)
    {
        var ones = Enumerable.Repeat(T.One, (int)shape.TotalLength).ToArray();
        return Create(ones, shape);
    }

    /// <summary>
    ///     Creates a volume with all elements initialized to zero.
    /// </summary>
    /// <param name="shape">The shape of the volume.</param>
    /// <returns>A volume filled with zeros.</returns>
    public static Volume<T> Zeros(Shape shape)
    {
        var zeros = Enumerable.Repeat(T.Zero, (int)shape.TotalLength).ToArray();
        return Create(zeros, shape);
    }

    public void ActivationGradient(Volume<T> input, Volume<T> outputGradient, ActivationType type, Volume<T> result)

    {
        throw new NotImplementedException();
    }

    public void Activation(ActivationType type, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void Dropout(float dropoutProb, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void DropoutGradient(Volume<T> dropoutInput, Volume<T> dropoutOutputGradient, float dropoutProb,
        Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void Pool(int width, int height, int horizontalPad, int verticalPad, int horizontalStride,
        int verticalStride, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void PoolGradient(Volume<T> volume, Volume<T> evaluate, int width, int height, int horizontalPad,
        int verticalPad, int horizontalStride, int verticalStride, Volume<T> inputGradient)
    {
        throw new NotImplementedException();
    }

    public void Reduce(TensorReduceOp add, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public T Get(int[] coordinates)
    {
        return this._storage.Get(coordinates);
    }

    public T Get(int w, int h, int c, int n)
    {
        return this._storage.Get(w, h, c, n);
    }

    public T Get(int w, int h, int c)
    {
        return this._storage.Get(w, h, c, 0);
    }

    public T Get(int w, int h)
    {
        return this._storage.Get(w, h, 0, 0);
    }

    public T Get(int i)
    {
        return this._storage.Get(i);
    }

    /// <summary>
    ///     Sets the value at a specific 1D coordinate in the volume.
    /// </summary>
    public void Set(int i, T value)
    {
        this.ThrowIfDisposed();
        this._storage.Set(0, 0, i, 0, value);
    }

    /// <summary>
    ///     Sets the value at a specific 2D coordinate in the volume.
    /// </summary>
    public void Set(int w, int h, T value)
    {
        this.ThrowIfDisposed();
        this._storage.Set(w, h, 0, 0, value);
    }

    /// <summary>
    ///     Sets the value at a specific 3D coordinate in the volume.
    /// </summary>
    public void Set(int w, int h, int c, T value)
    {
        this.ThrowIfDisposed();
        this._storage.Set(w, h, c, 0, value);
    }

    /// <summary>
    ///     Sets the value at a specific 4D coordinate in the volume.
    /// </summary>
    public void Set(int w, int h, int c, int n, T value)
    {
        this.ThrowIfDisposed();
        this._storage.Set(w, h, c, n, value);
    }

    /// <summary>
    ///     Sets the value at a specific n-dimensional coordinate in the volume.
    /// </summary>
    public void Set(int[] coordinates, T value)
    {
        this.ThrowIfDisposed();
        this._storage.Set(coordinates, value);
    }

    public void Sqrt(Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void Softmax(Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void Divide(Volume<T> right, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void Multiply(Volume<T> right, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void SoftmaxGradient(Volume<T> evaluate, Volume<T> inputGradient)
    {
        throw new NotImplementedException();
    }

    public void Exp(Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void MatMultiply(Volume<T> right, Volume<T> result)
    {
        // Ensure inputs are not null
        if (right == null) throw new ArgumentNullException(nameof(right));
        if (result == null) throw new ArgumentNullException(nameof(result));

        // Validate shapes for matrix multiplication
        if (this.Shape.Dimensions[1] != right.Shape.Dimensions[0] ||
            this.Shape.Dimensions[0] != result.Shape.Dimensions[0] ||
            right.Shape.Dimensions[1] != result.Shape.Dimensions[1])
        {
            throw new ArgumentException("Shapes are not compatible for matrix multiplication.");
        }

        // Delegate to the storage implementation
        this._storage.MatMultiply(right._storage, result._storage);
    }
    
    public void Negate(Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void Power(Volume<T> volume, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void Extract(int length, int offset, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void LeakyRelu(T alpha, Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public void LeakyReluGradient(Volume<T> derivative, Volume<T> result, T alpha)
    {
        throw new NotImplementedException();
    }

    public void Transpose(Volume<T> result)
    {
        throw new NotImplementedException();
    }

    public T[] ToArray()
    {
        this.ThrowIfDisposed(); // Ensure the object is not disposed.
        return this._storage.GetData(); // Fetch the data from the storage.
    }

    public static Volume<T> operator *(Volume<T> volume, T scalar)
    {
        if (volume == null)
        {
            throw new ArgumentNullException(nameof(volume));
        }

        // Delegate the operation to the storage layer
        var resultStorage = volume._storage.MultiplyByScalar(scalar);
        return new Volume<T>(resultStorage, volume.Shape);
    }

    public static Volume<T> operator *(T scalar, Volume<T> volume)
    {
        // Delegate to the main implementation
        return volume * scalar;
    }

    public static Volume<T> operator -(Volume<T> left, Volume<T> right)
    {
        if (left == null)
        {
            throw new ArgumentNullException(nameof(left));
        }

        if (right == null)
        {
            throw new ArgumentNullException(nameof(right));
        }

        if (!left.Shape.Equals(right.Shape))
        {
            throw new ArgumentException("Volumes must have the same shape to subtract.");
        }

        // Use CreateEmpty to create a result volume with the same shape and device as the left operand
        var result = CreateEmpty(left.Shape, left._storage.Device);

        // Perform the subtraction
        left._storage.Subtract(right._storage, result._storage);

        return result;
    }
}