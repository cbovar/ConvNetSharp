using System;

namespace ConvNetSharp.Volume
{
    public abstract class VolumeStorage<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected VolumeStorage(Shape shape)
        {
            this.Shape = new Shape(shape);
        }

        public Shape Shape { get; set; }

        public abstract void Clear();

        public abstract void CopyFrom(VolumeStorage<T> source);

        public abstract T Get(int[] coordinates);

        public abstract T Get(int w, int h, int c, int n);

        public abstract T Get(int w, int h, int c);

        public abstract T Get(int w, int h);

        public abstract T Get(int i);

        public void Map(Func<T, T, T> f, VolumeStorage<T> other, VolumeStorage<T> result)
        {
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                result.Set(i, f(this.Get(i), other.Get(i)));
            }
        }

        public void Map(Func<T, T> f, VolumeStorage<T> result)
        {
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                result.Set(i, f(this.Get(i)));
            }
        }

        public void Map(Func<T, int, T> f, VolumeStorage<T> result)
        {
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                result.Set(i, f(this.Get(i), i));
            }
        }

        /// <summary>
        ///     Implement broadcast
        /// </summary>
        public void MapEx(Func<T, T, T> f, VolumeStorage<T> other, VolumeStorage<T> result)
        {
            var big = this;
            var small = other;

            if (small.Shape.TotalLength > big.Shape.TotalLength)
            {
                big = other;
                small = this;
            }
            else if (small.Shape.TotalLength == big.Shape.TotalLength)
            {
                if (!small.Shape.Equals(big.Shape))
                {
                    throw new ArgumentException("Volumes have the same total number of dimensions but have different shapes");
                }

                // No broadcast to do here -> we switch to non-broacast implem
                this.Map(f, other, result);
                return;
            }

            var w = big.Shape.Dimensions[0];
            var h = big.Shape.Dimensions[1];
            var C = big.Shape.Dimensions[2];
            var N = big.Shape.Dimensions[3];

            var otherWIsOne = small.Shape.Dimensions[0] == 1;
            var otherHIsOne = small.Shape.Dimensions[1] == 1;
            var otherCIsOne = small.Shape.Dimensions[2] == 1;
            var otherNIsOne = small.Shape.Dimensions[3] == 1;

            for (var n = 0; n < N; n++)
            {
                for (var c = 0; c < C; c++)
                {
                    for (var j = 0; j < h; j++)
                    {
                        for (var i = 0; i < w; i++)
                        {
                            result.Set(i, j, c, n,
                                f(big.Get(i, j, c, n),
                                    small.Get(otherWIsOne ? 0 : i, otherHIsOne ? 0 : j, otherCIsOne ? 0 : c,
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
                this.Set(i, f(this.Get(i)));
            }
        }

        public void MapInplace(Func<T, T, T> f, VolumeStorage<T> other)
        {
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                this.Set(i, f(this.Get(i), other.Get(i)));
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