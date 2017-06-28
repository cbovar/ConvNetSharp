using System;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Layers
{
    public class ConvLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _filterCount;
        private readonly int _height;
        private readonly int _width;
        private Variable<T> _bias;

        public ConvLayer(int width, int height, int filterCount)
        {
            this._width = width;
            this._height = height;
            this._filterCount = filterCount;
        }

        public int Stride { get; set; } = 1;

        public int Pad { get; set; }

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);

            var cns = ConvNetSharp<T>.Instance;

            using (ConvNetSharp<T>.Instance.Scope($"ConvLayer_{this.Id}"))
            {
                this._bias = cns.Variable(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, this._filterCount)), "bias");
                this.Op = cns.Conv(parent.Op, this._width, this._height, this._filterCount, this.Stride, this.Pad) + this._bias;
            }
        }
    }
}