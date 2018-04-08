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

            var graph = parent.Op.Graph;
            this.Y = graph.PlaceHolder("Y");
            this.Op = graph.Softmax(parent.Op);
            this.Cost = graph.CrossEntropyLoss(this.Op, this.Y);
        }
    }
}