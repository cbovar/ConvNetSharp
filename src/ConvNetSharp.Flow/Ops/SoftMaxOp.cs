using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class SoftMaxOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;
        private long _lastGradientComputeStep = -1;

        public SoftMaxOp(Op<T> x)
        {
            this._x = x;
            AddParent(x);
        }

        public override string Representation => "Softmax";
        public Volume<T> InputGradient { get; set; }

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(new SoftMaxGradientOp<T>(this));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var y = this._x.Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, y.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(y.Shape);
            }

            y.DoSoftMax(this.Result);

            return this.Result;
        }

        public void EvaluateGradient(Session<T> session)
        {
            if (this._lastGradientComputeStep == session.Step)
            {
                return;
            }
            this._lastGradientComputeStep = session.Step;

            var x = this._x.Evaluate(session);

            if (this.InputGradient == null || !Equals(x.Shape, this.InputGradient.Shape))
            {
                this.InputGradient = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoSoftMaxGradient(this.Result, this.Derivate.Evaluate(session), this.InputGradient);
        }

        public override string ToString()
        {
            return $"SoftMax({this._x})";
        }
    }
}