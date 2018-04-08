using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers
{
    public class SoftmaxLayer<T> : LayerBase<T>, ILastLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        public PlaceHolder<T> Y { get; private set; }

        public Op<T> Cost { get; private set; }

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);

            var cns = parent.Op.Graph;
            this.Y = cns.PlaceHolder("Y");
            this.Op = cns.Softmax(parent.Op);
            this.Cost = cns.CrossEntropyLoss(this.Op, this.Y);
        }
    }
}