using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = a + b
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Add<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _left;
        private readonly Op<T> _right;

        public Add(Op<T> left, Op<T> right)
        {
            this._left = left;
            this._right = right;

            AddParent(left);
            AddParent(right);
        }

        public override string Representation => "+";

        public override void Differentiate()
        {
            this._left.RegisterDerivate(this.Derivate);
            this._right.RegisterDerivate(this.Derivate);
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

            var left = this._left.Evaluate(session);
            var right = this._right.Evaluate(session);

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
            return $"{this._left} + {this._right}";
        }
    }
}