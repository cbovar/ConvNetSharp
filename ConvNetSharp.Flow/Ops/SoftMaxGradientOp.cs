using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class SoftMaxGradientOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly SoftmaxOp<T> _softmaxOp;

        public SoftMaxGradientOp(SoftmaxOp<T> softmaxOp)
        {
            this._softmaxOp = softmaxOp;
            this.AddParent(softmaxOp);
        }

        public override string Representation => "SoftMaxGradient";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            this._softmaxOp.EvaluateGradient(session);
            return this._softmaxOp.InputGradient;
        }
    }
}