using System;
using System.Numerics;

namespace ConvNetSharp.Volume;

public class NcwhVolumeStorage<T> : VolumeStorage<T> where T : INumber<T>, IMinMaxValue<T>
{
    private readonly int _dim0, _dim0Dm1, _dim0Dm1Dm2;
    private Memory<T> _storage;

    public NcwhVolumeStorage(Shape shape) : base(shape)
    {
        this._storage = new T[shape.TotalLength].AsMemory();
        (this._dim0, this._dim0Dm1, this._dim0Dm1Dm2) = InitializeDimensions(shape);
    }

    public NcwhVolumeStorage(T[] array, Shape shape) : base(shape)
    {
        this._storage = array.AsMemory(); // Avoid copying unless necessary
        this.Shape.GuessUnknownDimension(this._storage.Length);
        (this._dim0, this._dim0Dm1, this._dim0Dm1Dm2) = InitializeDimensions(shape);
    }

    // Used by dropout layer
    public bool[]? Dropped { get; set; }

    private static (int dim0, int dim0Dm1, int dim0Dm1Dm2) InitializeDimensions(Shape shape)
    {
        var dim0 = shape.Dimensions[0];
        var dim1 = shape.Dimensions[1];
        var dim2 = shape.Dimensions[2];
        return (dim0, dim0 * dim1, dim0 * dim1 * dim2);
    }

    public override void Clear()
    {
        this._storage.Span.Clear();
    }

    public override void CopyFrom(VolumeStorage<T> source)
    {
        if (source is NcwhVolumeStorage<T> src && !ReferenceEquals(this, src))
        {
            if (this.Shape.TotalLength != src.Shape.TotalLength)
            {
                throw new ArgumentException(
                    $"Origin and destination volume should have the same number of weights ({this.Shape.TotalLength} != {src.Shape.TotalLength}).");
            }

            src._storage.Span.CopyTo(this._storage.Span);
        }
    }

    public override T Get(int[] coordinates)
    {
        var length = coordinates.Length;
        return this.Get(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0,
            length > 3 ? coordinates[3] : 0);
    }

    public override T Get(int w, int h, int c, int n)
    {
        return this._storage.Span[w + h * this._dim0 + c * this._dim0Dm1 + n * this._dim0Dm1Dm2];
    }

    public override T Get(int w, int h, int c)
    {
        return this._storage.Span[w + h * this._dim0 + c * this._dim0Dm1];
    }

    public override T Get(int w, int h)
    {
        return this._storage.Span[w + h * this._dim0];
    }

    public override T Get(int i)
    {
        return this._storage.Span[i];
    }

    public NcwhVolumeStorage<T> ReShape(Shape shape)
    {
        var storage = new NcwhVolumeStorage<T>(shape) { _storage = this._storage };
        return storage;
    }

    public override void Set(int[] coordinates, T value)
    {
        var length = coordinates.Length;
        this.Set(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0,
            length > 3 ? coordinates[3] : 0, value);
    }

    public override void Set(int w, int h, int c, int n, T value)
    {
        this._storage.Span[w + h * this._dim0 + c * this._dim0Dm1 + n * this._dim0Dm1Dm2] = value;
    }

    public override void Set(int w, int h, int c, T value)
    {
        this._storage.Span[w + h * this._dim0 + c * this._dim0Dm1] = value;
    }

    public override void Set(int w, int h, T value)
    {
        this._storage.Span[w + h * this._dim0] = value;
    }

    public override void Set(int i, T value)
    {
        this._storage.Span[i] = value;
    }

    public override T[] ToArray()
    {
        return this._storage.ToArray();
    }

    public override void MapInPlace(Func<T, T> f)
    {
        var span = this._storage.Span; // Direct access to memory
        for (var i = 0; i < this.Shape.TotalLength; i++)
        {
            span[i] = f(span[i]);
        }
    }

    public override void MapInPlace(Func<T, T, T> f, VolumeStorage<T> other)
    {
        if (other is NcwhVolumeStorage<T> otherStorage)
        {
            var span = this._storage.Span;
            var otherSpan = otherStorage._storage.Span;

            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                span[i] = f(span[i], otherSpan[i]);
            }
        }
        else
        {
            // Fallback to base behavior for other VolumeStorage types
            base.MapInPlace(f, other);
        }
    }
}