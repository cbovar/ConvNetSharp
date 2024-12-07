namespace ConvNetSharp.V2.Tests;

internal class MockVolumeStorage<T>(T[] data, Shape shape) : IVolumeStorage<T>
    where T : struct
{
    private readonly T[] _data = data;

    public Shape Shape { get; } = shape;

    public T[] GetData()
    {
        return this._data;
    }

    public T Get(int[] coordinates)
    {
        this.ValidateCoordinates(coordinates);

        var index = this.CalculateFlatIndex(coordinates);
        return this._data[index];
    }

    public T Get(int w, int h, int c, int n)
    {
        return this.Get([w, h, c, n]);
    }

    public T Get(int w, int h, int c)
    {
        return this.Get([w, h, c, 0]);
    }

    public T Get(int w, int h)
    {
        return this.Get([w, h, 0, 0]);
    }

    public T Get(int i)
    {
        if (i < 0 || i >= this._data.Length)
        {
            throw new IndexOutOfRangeException($"Index {i} is out of range.");
        }

        return this._data[i];
    }

    public void Set(int[] coordinates, T value)
    {
        this.ValidateCoordinates(coordinates);

        var index = this.CalculateFlatIndex(coordinates);
        this._data[index] = value;
    }

    public void Set(int w, int h, int c, int n, T value)
    {
        this.Set([w, h, c, n], value);
    }

    public void Set(int w, int h, int c, T value)
    {
        this.Set([w, h, c, 0], value);
    }

    public void Set(int w, int h, T value)
    {
        this.Set([w, h, 0, 0], value);
    }

    public void Set(int i, T value)
    {
        if (i < 0 || i >= this._data.Length)
        {
            throw new IndexOutOfRangeException($"Index {i} is out of range.");
        }

        this._data[i] = value;
    }

    public void ApplyOperation(Func<T, T> operation)
    {
        for (var i = 0; i < this._data.Length; i++)
        {
            this._data[i] = operation(this._data[i]);
        }
    }

    public void Add(IVolumeStorage<T> other, IVolumeStorage<T> result)
    {
        if (other is not MockVolumeStorage<T> otherMock || result is not MockVolumeStorage<T> resultMock)
        {
            throw new ArgumentException("Both other and result must be of type MockVolumeStorage<T>.");
        }

        if (this.Shape.TotalLength != otherMock.Shape.TotalLength ||
            this.Shape.TotalLength != resultMock.Shape.TotalLength)
        {
            throw new ArgumentException("Shapes of all storages must match.");
        }

        for (var i = 0; i < this._data.Length; i++)
        {
            resultMock._data[i] = (dynamic)this._data[i] + (dynamic)otherMock._data[i];
        }
    }

    public void Add(IVolumeStorage<T> result)
    {
        throw new NotImplementedException();
    }

    public void Clear()
    {
        Array.Fill(this._data, default);
    }

    public void Convolution(IVolumeStorage<T> filterStorage, int xpad, int ypad, int stride, IVolumeStorage<T> outputStorage)
    {
        throw new NotImplementedException();
    }

    public void Tile(IVolumeStorage<T> reps, IVolumeStorage<T> result)
    {
        throw new NotImplementedException();
    }

    public IVolumeStorage<T> MultiplyByScalar(T scalar, bool inPlace = false)
    {
        throw new NotImplementedException();
    }

    private int CalculateFlatIndex(int[] coordinates)
    {
        var index = 0;
        var stride = 1;

        for (var i = this.Shape.Dimensions.Length - 1; i >= 0; i--)
        {
            index += coordinates[i] * stride;
            stride *= this.Shape.Dimensions[i];
        }

        return index;
    }

    private void ValidateCoordinates(int[] coordinates)
    {
        if (coordinates.Length != this.Shape.Dimensions.Length)
        {
            throw new ArgumentException("Coordinate dimensions do not match the storage shape.");
        }

        for (var i = 0; i < coordinates.Length; i++)
        {
            if (coordinates[i] < 0 || coordinates[i] >= this.Shape.Dimensions[i])
            {
                throw new ArgumentOutOfRangeException($"Coordinate {i} is out of range.");
            }
        }
    }
}