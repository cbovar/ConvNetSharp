using System;

namespace ConvNetSharp.Volume
{
    public class NcwhVolumeStorage<T> : VolumeStorage<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Layout _layout;
        private T[] _storage;

        public NcwhVolumeStorage(Shape shape) : base(shape)
        {
            this._storage = new T[shape.TotalLength];
            this._layout = new Layout(this.Shape.Dimensions); // TODO: not hardcode layout 

#if PERF
            Console.WriteLine($"Alloc {this.Shape.PrettyPrint()}({this.Shape.TotalLength} units)");
#endif
        }

        public NcwhVolumeStorage(T[] array, Shape shape) : base(shape)
        {
            this._storage = (T[])array.Clone();
            this.Shape.GuessUnkownDimension(this._storage.Length);
            this._layout = new Layout(this.Shape.Dimensions); // TODO: not hardcode layout 
        }

        public override void Clear()
        {
            Array.Clear(this._storage, 0, this._storage.Length);
        }

        public override bool Equals(object obj)
        {
            return Equals(obj as VolumeStorage<T>);
        }

        public override bool Equals(VolumeStorage<T> other)
        {
            throw new NotImplementedException();
        }

        public override T Get(int[] coordinates)
        {
            var i = this._layout.IndexFromCoordinates(coordinates);
            return this._storage[i];
        }

        public override T Get(int w, int h, int c, int n)
        {
            var i = this._layout.IndexFromCoordinates(w, h, c, n);
            return this._storage[i];
        }

        public override T Get(int w, int h, int c)
        {
            var i = this._layout.IndexFromCoordinates(w, h, c);
            return this._storage[i];
        }

        public override T Get(int w, int h)
        {
            var i = this._layout.IndexFromCoordinates(w, h);
            return this._storage[i];
        }

        public override T Get(int w)
        {
            var i = this._layout.IndexFromCoordinates(w);
            return this._storage[i];
        }

        public NcwhVolumeStorage<T> ReShape(Shape shape)
        {
            var storage = new NcwhVolumeStorage<T>(shape);
            storage._storage = this._storage;
            return storage;
        }

        public override void Set(int[] coordinates, T value)
        {
            var i = this._layout.IndexFromCoordinates(coordinates);
            this._storage[i] = value;
        }

        public override void Set(int w, int h, int c, int n, T value)
        {
            var i = this._layout.IndexFromCoordinates(w, h, c, n);
            this._storage[i] = value;
        }

        public override void Set(int w, int h, int c, T value)
        {
            var i = this._layout.IndexFromCoordinates(w, h, c);
            this._storage[i] = value;
        }

        public override void Set(int w, int h, T value)
        {
            var i = this._layout.IndexFromCoordinates(w, h);
            this._storage[i] = value;
        }

        public override void Set(int w, T value)
        {
            var i = this._layout.IndexFromCoordinates(w);
            this._storage[i] = value;
        }

        public override T[] ToArray()
        {
            return (T[])this._storage.Clone();
        }
    }
}