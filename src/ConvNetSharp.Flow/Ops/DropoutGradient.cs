using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class DropoutGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Dropout<T> _dropout;
        private Shape _lastInputShape;

        public DropoutGradient(Dropout<T> dropout, Op<T> derivate)
        {
            this._dropout = dropout;
            AddParent(dropout);
            AddParent(derivate);
        }

        public override string Representation => "DropoutGradient";

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

            var dropoutOutput = this._dropout.Evaluate(session);
            var dropoutInput = this._dropout.Parents[0].Evaluate(session);
            var dropoutInputGradient = this._dropout.Derivate.Evaluate(session);

            if (this.Result == null || !Equals(this._lastInputShape, dropoutInput.Shape))
            {
                this._lastInputShape = new Shape(dropoutInput.Shape);

                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(dropoutInput.Shape);
            }


            dropoutOutput.DoDropoutGradient(dropoutInput, this.Result, dropoutInputGradient, this._dropout.DropoutProbability);

            return this.Result;
        }
    }
}