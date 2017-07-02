using System;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Layers
{
    public class FullyConnLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int _neuronCount;
        private bool _initialized;
        private T _biasPref;
        private Variable<T> _bias;

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

        public override void AcceptParent(LayerBase<T> parent)
        {
            base.AcceptParent(parent);

            var cns = ConvNetSharp<T>.Instance;

            using (cns.Scope($"FullConnLayer_{this.Id}"))
            {
                this._bias = cns.Variable(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, this._neuronCount, 1)), "Bias");
                this.Op = cns.Conv(cns.Reshape(parent.Op, new Shape(1, 1, -1, Shape.Keep)), 1, 1, this._neuronCount) + this._bias;
            }

            this._initialized = true;
        }
    }
}