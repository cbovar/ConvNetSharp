using System;

namespace ConvNetSharp.Flow.Layers
{
    public class LeakyReluLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public LeakyReluLayer(T alpha)
        {
            this.Alpha = alpha;
        }

        public T Alpha { get; set; }

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);
            this.Op = parent.Op.Graph.LeakyRelu(parent.Op, this.Alpha);
        }
    }
}