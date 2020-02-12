using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class ActivationGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public ActivationGradient(ConvNetSharp<T> graph, Op<T> input, Op<T> output, Op<T> outputGradient, ActivationType type) : base(graph)
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
                this.Result?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var input = this.Parents[0].Evaluate(session);
            var output = this.Parents[1].Evaluate(session);
            var outputGradient = this.Parents[2].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, input.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(input.Shape);
            }

            output.ActivationGradient(input, outputGradient, this.Type, this.Result);

            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"{this.Type} Gradient({this.Parents[0]})";
        }
    }
}