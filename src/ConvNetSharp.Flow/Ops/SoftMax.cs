using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Softmax<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private long _lastGradientComputeStep = -1;

        public Softmax(Op<T> x)
        {
            AddParent(x);
        }

        public Softmax(Dictionary<string, object> data)
        {
        }

        public override string Representation => "Softmax";

        public Volume<T> InputGradient { get; set; }

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(new SoftmaxGradient<T>(this));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, x.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoSoftmax(this.Result);

            return this.Result;
        }

        public void EvaluateGradient(Session<T> session)
        {
            if (this._lastGradientComputeStep == session.Step)
            {
                return;
            }
            this._lastGradientComputeStep = session.Step;

            var x = this.Parents[0].Evaluate(session);

            if (this.InputGradient == null || !Equals(x.Shape, this.InputGradient.Shape))
            {
                this.InputGradient = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            if (this.Derivate != null)
            {
                x.DoSoftmaxGradient(this.Derivate.Evaluate(session), this.InputGradient);
            }
        }

        public override string ToString()
        {
            return $"Softmax({this.Parents[0]})";
        }
    }
}