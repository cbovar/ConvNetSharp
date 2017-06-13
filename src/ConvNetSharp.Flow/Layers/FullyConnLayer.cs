using System;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Layers
{
    public class FullyConnLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _neuronCount;

        private Variable<T> _bias;

        public FullyConnLayer(int neuronCount)
        {
            this._neuronCount = neuronCount;
        }

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);

            using (ConvNetSharp<T>.Instance.Scope($"FullyConnLayer{this.Id}"))
            {
                this._bias = ConvNetSharp<T>.Instance.Variable(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, this._neuronCount, 1)), "bias");
                this.Op = ConvNetSharp<T>.Instance.Conv(parent.Op, 1, 1, this._neuronCount) + this._bias;
            }
        }
    }
}