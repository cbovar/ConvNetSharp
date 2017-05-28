using System;

namespace ConvNetSharp.Volume
{
    public abstract class VolumeStorage<T> : IEquatable<VolumeStorage<T>>, IDisposable
        where T : struct, IEquatable<T>, IFormattable
    {
        protected VolumeStorage(Shape shape)
        {
            this.Shape = new Shape(shape);
        }

        public Shape Shape { get; set; }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public abstract bool Equals(VolumeStorage<T> other);

        public abstract void Clear();

        protected virtual void Dispose(bool disposing)
        {
        }

        public abstract T Get(int[] coordinates);

        public abstract T Get(int w, int h, int c, int n);

        public abstract T Get(int w, int h, int c);

        public abstract T Get(int w, int h);

        public abstract T Get(int i);

        public void Map(Func<T, T, T> f, VolumeStorage<T> other, VolumeStorage<T> result)
        {
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                result.Set(i, f(Get(i), other.Get(i)));
            }
        }

        public void Map(Func<T, T> f, VolumeStorage<T> result)
        {
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                result.Set(i, f(Get(i)));
            }
        }

        public void MapEx(Func<T, T, T> f, VolumeStorage<T> other, VolumeStorage<T> result)
        {
            var w = this.Shape.GetDimension(0);
            var h = this.Shape.GetDimension(1);
            var C = this.Shape.GetDimension(2);
            var N = this.Shape.GetDimension(3);

            var otherWIsOne = other.Shape.GetDimension(0) == 1;
            var otherHIsOne = other.Shape.GetDimension(1) == 1;
            var otherCIsOne = other.Shape.GetDimension(2) == 1;
            var otherNIsOne = other.Shape.GetDimension(3) == 1;

            for (var n = 0; n < N; n++)
            {
                for (var c = 0; c < C; c++)
                {
                    for (var j = 0; j < h; j++)
                    {
                        for (var i = 0; i < w; i++)
                        {
                            result.Set(i, j, c, n,
                                f(Get(i, j, c, n),
                                    other.Get(otherWIsOne ? 0 : i, otherHIsOne ? 0 : j, otherCIsOne ? 0 : c,
                                        otherNIsOne ? 0 : n)));
                        }
                    }
                }
            }
        }

        public void MapInplace(Func<T, T> f)
        {
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                Set(i, f(Get(i)));
            }
        }

        public void MapInplace(Func<T, T, T> f, VolumeStorage<T> other)
        {
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                Set(i, f(Get(i), other.Get(i)));
            }
        }

        public abstract void Set(int[] coordinates, T value);

        public abstract void Set(int w, int h, int c, int n, T value);

        public abstract void Set(int w, int h, int c, T value);

        public abstract void Set(int w, int h, T value);

        public abstract void Set(int i, T value);

        public abstract T[] ToArray();
    }
}