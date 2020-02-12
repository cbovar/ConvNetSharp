using System;

namespace ConvNetSharp.Volume
{
    public class NcwhVolumeStorage<T> : VolumeStorage<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _dim0, _dim0Dm1, _dim0Dm1Dm2;
        private T[] _storage;

        public NcwhVolumeStorage(Shape shape) : base(shape)
        {
            this._storage = new T[shape.TotalLength];

            this._dim0 = this.Shape.Dimensions[0];
            var dim1 = this.Shape.Dimensions[1];
            var dim2 = this.Shape.Dimensions[2];
            this._dim0Dm1 = this._dim0 * dim1;
            this._dim0Dm1Dm2 = this._dim0 * dim1 * dim2;

#if PERF
            Console.WriteLine($"Alloc {this.Shape.PrettyPrint()}({this.Shape.TotalLength} units)");
#endif
        }

        public NcwhVolumeStorage(T[] array, Shape shape) : base(shape)
        {
            this._storage = (T[])array.Clone();
            this.Shape.GuessUnknownDimension(this._storage.Length);

            this._dim0 = this.Shape.Dimensions[0];
            var dim1 = this.Shape.Dimensions[1];
            var dim2 = this.Shape.Dimensions[2];
            this._dim0Dm1 = this._dim0 * dim1;
            this._dim0Dm1Dm2 = this._dim0 * dim1 * dim2;
        }

        // Used by dropout layer
        public bool[] Dropped { get; set; }

        public override void Clear()
        {
            Array.Clear(this._storage, 0, this._storage.Length);
        }

        public override void CopyFrom(VolumeStorage<T> source)
        {
            var src = source as NcwhVolumeStorage<T>;

            if (!ReferenceEquals(this, src))
            {
                if (this.Shape.TotalLength != src.Shape.TotalLength)
                {
                    throw new ArgumentException($"origin and destination volume should have the same number of weight ({this.Shape.TotalLength} != {src.Shape}).");
                }

                Array.Copy(src._storage, this._storage, this._storage.Length);
            }
        }

        public override T Get(int[] coordinates)
        {
            var length = coordinates.Length;
            return this.Get(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0, length > 3 ? coordinates[3] : 0);
        }

        public override T Get(int w, int h, int c, int n)
        {
            return this._storage[w + h * this._dim0 + c * this._dim0Dm1 + n * this._dim0Dm1Dm2];
        }

        public override T Get(int w, int h, int c)
        {
            return
                this._storage[w + h * this._dim0 + c * this._dim0Dm1];
        }

        public override T Get(int w, int h)
        {
            return this._storage[w + h * this._dim0];
        }

        public override T Get(int i)
        {
            return this._storage[i];
        }

        public NcwhVolumeStorage<T> ReShape(Shape shape)
        {
            var storage = new NcwhVolumeStorage<T>(shape) { _storage = this._storage };
            return storage;
        }

        public override void Set(int[] coordinates, T value)
        {
            var length = coordinates.Length;
            this.Set(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0, length > 3 ? coordinates[3] : 0, value);
        }

        public override void Set(int w, int h, int c, int n, T value)
        {
            this._storage[w + h * this._dim0 + c * this._dim0Dm1 + n * this._dim0Dm1Dm2] = value;
        }

        public override void Set(int w, int h, int c, T value)
        {
            this._storage[w + h * this._dim0 + c * this._dim0Dm1] = value;
        }

        public override void Set(int w, int h, T value)
        {
            this._storage[w + h * this._dim0] = value;
        }

        public override void Set(int i, T value)
        {
            this._storage[i] = value;
        }

        public override T[] ToArray()
        {
            return (T[])this._storage.Clone();
        }
    }
}