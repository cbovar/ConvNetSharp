using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers
{
    public class PoolLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _height;
        private readonly int _width;
        private bool _initialized;
        private int _pad;
        private Pool<T> _pool;
        private int _stride = 1;

        public PoolLayer(int width, int height)
        {
            this._width = width;
            this._height = height;
        }

        public int Stride
        {
            get => this._stride;
            set
            {
                this._stride = value;
                if (this._initialized)
                {
                    this._pool.HorizontalStride = value;
                    this._pool.VerticalStride = value;
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
                    this._pool.HorizontalPad = value;
                    this._pool.VerticalPad = value;
                }
            }
        }

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);
            this._pool = parent.Op.Graph.Pool(parent.Op, this._width, this._height, this.Pad, this.Pad, this.Stride, this.Stride);
            this.Op = this._pool;

            this._initialized = true;
        }
    }
}