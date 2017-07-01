using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class SoftmaxGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Softmax<T> _softmax;

        public SoftmaxGradient(Softmax<T> softmax)
        {
            this._softmax = softmax;
            this.AddParent(softmax);
        }

        public override string Representation => "SoftmaxGradient";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            this._softmax.EvaluateGradient(session);

            this.Result = this._softmax.InputGradient;
            return this.Result;
        }
    }
}