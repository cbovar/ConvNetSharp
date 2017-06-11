using System;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Layers
{
    public class FullyConnLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _neuronCount;

        private readonly Variable<T> _bias;

        public FullyConnLayer(int neuronCount)
        {
            this._neuronCount = neuronCount;
            this._bias = new Variable<T>(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, neuronCount, 1)), "bias");
        }

        public override void AcceptParent(Op<T> parent)
        {
            this.Op = ConvNetSharp<T>.Conv(parent, 1, 1, this._neuronCount) + this._bias;
        }
    }
}