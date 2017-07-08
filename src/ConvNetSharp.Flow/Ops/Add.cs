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
        public Add(Dictionary<string, object> data)
        {
        }

        public Add(Op<T> left, Op<T> right)
        {
            AddParent(left);
            AddParent(right);
        }

        public override string Representation => "+";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(ConvNetSharp<T>.Instance.Sum(this.Derivate, ConvNetSharp<T>.Instance.Shape(this.Parents[0])));
            this.Parents[1].RegisterDerivate(ConvNetSharp<T>.Instance.Sum(this.Derivate, ConvNetSharp<T>.Instance.Shape(this.Parents[1])));
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
                return this.Result;
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

            left.DoAdd(right, this.Result);

            return this.Result;
        }

        public override string ToString()
        {
            return $"{this.Parents[0]} + {this.Parents[1]}";
        }
    }
}