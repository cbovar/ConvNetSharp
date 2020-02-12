using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Matrix multiplication
    ///     y = left * right
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class MatMult<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public MatMult(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public MatMult(ConvNetSharp<T> graph, Op<T> left, Op<T> right) : base(graph)
        {
            this.AddParent(left);
            this.AddParent(right);
        }

        public override string Representation => "*";

        public override void Differentiate()
        {
            // dA = GBᵀ, dB = AᵀG
            this.Parents[0].RegisterDerivate(this.Graph.MatMult(this.Derivate, this.Graph.Transpose(this.Parents[1])));
            this.Parents[1].RegisterDerivate(this.Graph.MatMult(this.Graph.Transpose(this.Parents[0]), this.Derivate));
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

            var left = this.Parents[0].Evaluate(session);
            var right = this.Parents[1].Evaluate(session);

            var shape = Volume<T>.ComputeMatMultiplyShape(left.Shape, right.Shape);

            if (this.Result == null || !Equals(this.Result.Shape, shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(shape);
            }

            left.MatMultiply(right, this.Result);

            return base.Evaluate(session);
        }

        public override string ToString()
        {
            var addParenthesis = this.Parents[0].Parents.Any();
            var leftStr = addParenthesis ? $"({this.Parents[0]})" : $"{this.Parents[0]}";

            addParenthesis = this.Parents[1].Parents.Any();
            var rightStr = addParenthesis ? $"({this.Parents[1]})" : $"{this.Parents[1]}";

            return $"{leftStr} * {rightStr}";
        }
    }
}