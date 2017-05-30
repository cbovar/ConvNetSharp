using System;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class ConvLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _filterCount;
        private readonly int _height;
        private readonly int _width;
        private readonly Variable<T> _bias;

        public ConvLayer(int width, int height, int filterCount)
        {
            this._width = width;
            this._height = height;
            this._filterCount = filterCount;
            this._bias = new Variable<T>(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, filterCount)), "bias");
        }

        public int Stride { get; set; } = 1;

        public int Pad { get; set; }

        public override void AcceptParent(Op<T> parent)
        {
            this.Op = ConvNetSharp<T>.Conv(parent, this._width, this._height, this._filterCount, this.Stride, this.Pad) + this._bias;
        }
    }
}