using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = a + b
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Add<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Add(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public Add(ConvNetSharp<T> graph, Op<T> left, Op<T> right) : base(graph)
        {
            this.AddParent(left);
            this.AddParent(right);
        }

        public override string Representation => "+";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(this.Graph.Sum(this.Derivate, this.Graph.Shape(this.Parents[0])));
            this.Parents[1].RegisterDerivate(this.Graph.Sum(this.Derivate, this.Graph.Shape(this.Parents[1])));
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

            var shape = right.Shape.TotalLength > left.Shape.TotalLength ? right.Shape : left.Shape;

            if (this.Result == null || !Equals(this.Result.Shape, shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(shape);
            }

            left.Add(right, this.Result);

            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"{this.Parents[0]} + {this.Parents[1]}";
        }
    }
}