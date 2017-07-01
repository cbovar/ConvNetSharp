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

            var cns = ConvNetSharp<T>.Instance;

            using (cns.Scope($"FullyConnLayer{this.Id}"))
            {
                this._bias = cns.Variable(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, this._neuronCount, 1)), "bias");
                this.Op = cns.Conv(cns.Reshape(parent.Op, new Shape(1, 1, -1, Shape.Keep)), 1, 1, this._neuronCount) + this._bias;
            }
        }
    }
}