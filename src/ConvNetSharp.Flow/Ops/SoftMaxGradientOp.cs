using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class SoftMaxGradientOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly SoftMaxOp<T> _softMaxOp;

        public SoftMaxGradientOp(SoftMaxOp<T> softMaxOp)
        {
            this._softMaxOp = softMaxOp;
            this.AddParent(softMaxOp);
        }

        public override string Representation => "SoftMaxGradient";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            this._softMaxOp.EvaluateGradient(session);
            return this._softMaxOp.InputGradient;
        }
    }
}