using System;

namespace ConvNetSharp.Flow.Layers
{
    public class DropoutLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly T _dropoutProbability;

        public DropoutLayer(T dropoutProbability)
        {
            this._dropoutProbability = dropoutProbability;
        }

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);
            this.Op = parent.Op.Graph.Dropout(parent.Op, this._dropoutProbability);
        }
    }
}