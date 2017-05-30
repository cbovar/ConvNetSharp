using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class ActivationGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> _result;

        public ActivationGradient(Op<T> input, Op<T> output, Op<T> outputGradient, ActivationType type)
        {
            this.AddParent(input);
            this.AddParent(output);
            this.AddParent(outputGradient);

            this.Type = type;
        }

        public ActivationType Type { get; set; }

        public override string Representation => $"{this.Type} Gradient";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this._result?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this.LastComputeStep == session.Step) return this._result;
            this.LastComputeStep = session.Step;

            var input = this.Parents[0].Evaluate(session);
            var output = this.Parents[1].Evaluate(session);
            var outputGradient = this.Parents[2].Evaluate(session);

            if (this._result == null || !Equals(this._result.Shape, input.Shape))
            {
                this._result?.Dispose();
                this._result = BuilderInstance<T>.Volume.SameAs(input.Shape);
            }

            output.DoActivationGradient(input, outputGradient, this._result, this.Type);

            return this._result;
        }
    }
}