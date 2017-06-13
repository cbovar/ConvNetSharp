using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = f(x) where f can be Sigmoid ,Relu,Tanh or ClippedRelu
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Activation<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> _result;
        private readonly Op<T> _x;

        public Activation(Op<T> x, ActivationType type)
        {
            this._x = x;
            AddParent(x);
            this.Type = type;
        }

        public ActivationType Type { get; set; }

        public override string Representation => $"{this.Type}";

        public override void Differentiate()
        {
            this._x.RegisterDerivate(new ActivationGradient<T>(this._x, this, this.Derivate, this.Type));
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this._result?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this.LastComputeStep == session.Step)
            {
                return this._result;
            }
            this.LastComputeStep = session.Step;

            var x = this._x.Evaluate(session);

            if (this._result == null || !Equals(this._result.Shape, x.Shape))
            {
                this._result?.Dispose();
                this._result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoActivation(this._result, this.Type);
            return this._result;
        }

        public override string ToString()
        {
            return $"{this.Type}({this._x})";
        }
    }
}