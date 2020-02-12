using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class DropoutGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Dropout<T> _dropout;
        private Shape _lastInputShape;

        public DropoutGradient(ConvNetSharp<T> graph, Dropout<T> dropout, Op<T> derivate) : base(graph)
        {
            this._dropout = dropout;
            this.AddParent(dropout);
            this.AddParent(derivate);
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
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var dropoutOutput = this._dropout.Evaluate(session);
            var dropoutInput = this._dropout.Parents[0].Evaluate(session);
            var dropoutOutputGradient = this._dropout.Derivate.Evaluate(session);
            var dropoutProb = this._dropout.DropoutProbability.Evaluate(session);

            if (this.Result == null || !Equals(this._lastInputShape, dropoutInput.Shape))
            {
                this._lastInputShape = new Shape(dropoutInput.Shape);

                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(dropoutInput.Shape);
            }

            dropoutOutput.DropoutGradient(dropoutInput, dropoutOutputGradient, dropoutProb, this.Result);

            return base.Evaluate(session);
        }
    }
}