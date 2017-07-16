using System;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Layers
{
    public class FullyConnLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _neuronCount;
        private Variable<T> _bias;
        private T _biasPref;
        private Convolution<T> _conv;
        private bool _initialized;

        public FullyConnLayer(int neuronCount)
        {
            this._neuronCount = neuronCount;
        }

        public T BiasPref
        {
            get { return this._biasPref; }
            set
            {
                this._biasPref = value;

                if (this._initialized)
                {
                    this._bias.Result = new T[this._neuronCount].Populate(this.BiasPref);
                }
            }
        }

        public Op<T> Filter => this._conv.Filter;

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);

            var cns = ConvNetSharp<T>.Instance;

            using (cns.Scope($"FullConnLayer_{this.Id}"))
            {
                this._bias = cns.Variable(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, this._neuronCount, 1)), "Bias");
                this._conv = cns.Conv(cns.Reshape(parent.Op, new Shape(1, 1, -1, Shape.Keep)), 1, 1, this._neuronCount);
                this.Op = this._conv + this._bias;
            }

            this._initialized = true;
        }
    }
}