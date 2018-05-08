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
        private Dense<T> _conv;
        private bool _initialized;

        public FullyConnLayer(int neuronCount)
        {
            this._neuronCount = neuronCount;
        }

        public T BiasPref
        {
            get => this._biasPref;
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

            var cns = parent.Op.Graph;

            using (cns.Scope($"FullConnLayer_{this.Id}"))
            {
                this._bias = cns.Variable(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, this._neuronCount, 1)), "Bias", true);
                this._conv = cns.Dense(cns.Reshape(parent.Op, new Shape(1, 1, -1, Shape.Keep)), this._neuronCount);
                this.Op = this._conv + this._bias;
            }

            this._initialized = true;
        }
    }
}