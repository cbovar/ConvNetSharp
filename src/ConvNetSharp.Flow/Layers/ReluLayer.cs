using System;

namespace ConvNetSharp.Flow.Layers
{
    public class ReluLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);
            this.Op = parent.Op.Graph.Relu(parent.Op);
        }
    }
}