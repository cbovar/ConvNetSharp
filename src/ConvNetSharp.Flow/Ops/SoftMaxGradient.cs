using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class SoftMaxGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly SoftMax<T> _softMax;

        public SoftMaxGradient(SoftMax<T> softMax)
        {
            this._softMax = softMax;
            this.AddParent(softMax);
        }

        public override string Representation => "SoftMaxGradient";

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

            this._softMax.EvaluateGradient(session);

            this.Result = this._softMax.InputGradient;
            return this.Result;
        }
    }
}