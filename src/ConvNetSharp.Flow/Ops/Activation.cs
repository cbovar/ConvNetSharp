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
                this.Result?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this.LastComputeStep == session.Step)
            {
                return this.Result;
            }
            this.LastComputeStep = session.Step;

            var x = this._x.Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, x.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoActivation(this.Result, this.Type);
            return this.Result;
        }

        public override string ToString()
        {
            return $"{this.Type}({this._x})";
        }
    }
}