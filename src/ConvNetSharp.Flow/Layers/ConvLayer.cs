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
        private T _biasPref;
        private Convolution<T> _conv;
        private bool _initialized;
        private int _pad;
        private int _stride = 1;

        public ConvLayer(int width, int height, int filterCount)
        {
            this._width = width;
            this._height = height;
            this._filterCount = filterCount;
        }

        public T BiasPref
        {
            get => this._biasPref;
            set
            {
                this._biasPref = value;

                if (this._initialized)
                {
                    this._bias.Result = new T[this._filterCount].Populate(this.BiasPref);
                }
            }
        }

        public int Stride
        {
            get => this._stride;
            set
            {
                this._stride = value;
                if (this._initialized)
                {
                    this._conv.Stride = value;
                }
            }
        }

        public int Pad
        {
            get => this._pad;
            set
            {
                this._pad = value;
                if (this._initialized)
                {
                    this._conv.Pad = value;
                }
            }
        }

        public Op<T> Filter => this._conv.Filter;

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);

            var cns = parent.Op.Graph;

            using (cns.Scope($"ConvLayer_{this.Id}"))
            {
                var content = new T[this._filterCount].Populate(this.BiasPref);
                this._bias = cns.Variable(BuilderInstance<T>.Volume.From(content, new Shape(1, 1, this._filterCount, 1)), "Bias", true);
                this._conv = cns.Conv(parent.Op, this._width, this._height, this._filterCount, this.Stride, this.Pad);
                this.Op = this._conv + this._bias;
            }

            this._initialized = true;
        }
    }
}