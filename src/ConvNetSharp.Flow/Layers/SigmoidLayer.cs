using System;

namespace ConvNetSharp.Flow.Layers
{
    public class SigmoidLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);
            this.Op = parent.Op.Graph.Sigmoid(parent.Op);
        }
    }
}