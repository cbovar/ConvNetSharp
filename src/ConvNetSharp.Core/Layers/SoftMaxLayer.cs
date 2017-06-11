using System;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Core.Layers
{
    public class SoftmaxLayer<T> : LayerBase<T>, ILastLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        public PlaceHolder<T> Y { get; set; } = new PlaceHolder<T>("Y");

        public Op<T> Cost { get; set; }

        public override void AcceptParent(Op<T> parent)
        {
            this.Op = ConvNetSharp<T>.Softmax(parent);

            this.Cost = ConvNetSharp<T>.CrossEntropyLoss(parent, this.Y);
        }
    }
}