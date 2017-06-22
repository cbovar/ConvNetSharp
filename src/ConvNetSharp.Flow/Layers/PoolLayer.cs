using System;

namespace ConvNetSharp.Flow.Layers
{
    public class PoolLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _height;
        private readonly int _width;

        public PoolLayer(int width, int height)
        {
            this._width = width;
            this._height = height;
        }

        public int Stride { get; set; } = 1;

        public int Pad { get; set; } = 0;

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);
            this.Op = ConvNetSharp<T>.Instance.Pool(parent.Op, this._width, this._height, this.Pad, this.Pad, this.Stride, this.Stride);
        }
    }
}