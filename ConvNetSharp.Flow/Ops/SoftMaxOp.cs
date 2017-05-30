using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class SoftmaxOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;
        private long _lastGradientComputeStep = -1;
        private Volume<T> _result;

        public SoftmaxOp(Op<T> x)
        {
            this._x = x;
            AddParent(x);
        }

        public override string Representation => "Softmax";
        public Volume<T> InputGradient { get; set; }

        public override void Differentiate()
        {
            this.Parents[0].Derivate = new SoftMaxGradientOp<T>(this);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this.LastComputeStep == session.Step)
            {
                return this._result;
            }
            this.LastComputeStep = session.Step;

            var y = this._x.Evaluate(session);

            if (this._result == null || !Equals(this._result.Shape, y.Shape))
            {
                this._result?.Dispose();
                this._result = BuilderInstance<T>.Volume.SameAs(y.Shape);
            }

            y.DoSoftmax(this._result);

            return this._result;
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

            x.DoSoftmaxGradient(this.Derivate.Evaluate(session), this.InputGradient);
        }
    }
}