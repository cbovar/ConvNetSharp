using System;

namespace ConvNetSharp.Volume
{
    public class NcwhVolumeStorage<T> : VolumeStorage<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _dim0, _dim0Dm1, _dim0Dm1Dm2;
        private readonly T[] _storage;

        public NcwhVolumeStorage(Shape shape) : base(shape)
        {
            this._storage = new T[shape.TotalLength];

            this._dim0 = this.Shape.GetDimension(0);
            var dim1 = this.Shape.GetDimension(1);
            var dim2 = this.Shape.GetDimension(2);
            this._dim0Dm1 = this._dim0 * dim1;
            this._dim0Dm1Dm2 = this._dim0 * dim1 * dim2;

#if PERF
            Console.WriteLine($"Alloc {this.Shape.PrettyPrint()}({this.Shape.TotalLength} units)");
#endif
        }

        public NcwhVolumeStorage(T[] array, Shape shape) : base(shape)
        {
            this._storage = array;
            this.Shape.GuessUnkownDimension(this._storage.Length);

            this._dim0 = this.Shape.GetDimension(0);
            var dim1 = this.Shape.GetDimension(1);
            var dim2 = this.Shape.GetDimension(2);
            this._dim0Dm1 = this._dim0 * dim1;
            this._dim0Dm1Dm2 = this._dim0 * dim1 * dim2;
        }

        public override void Clear()
        {
            Array.Clear(this._storage, 0, this._storage.Length);
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

        public override void Set(int w, int h, int c, int n, T value)
        {
            this._storage[
                w + h * this._dim0 + c * this._dim0Dm1 + n * this._dim0Dm1Dm2] = value;
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
            return this._storage;
        }
    }
}