using System;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Element wise multiplication
    ///     y = left * right
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Mult<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _left;
        private readonly Op<T> _right;

        public Mult(Op<T> left, Op<T> right)
        {
            this._left = left;
            this._right = right;

            AddParent(left);
            AddParent(right);
        }

        public override string Representation => "*";

        public override void Differentiate()
        {
            this._left.RegisterDerivate(this.Derivate * this._right);
            this._right.RegisterDerivate(this.Derivate * this._left);
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

            left.DoMultiply(right, this.Result);

            return this.Result;
        }

        public override string ToString()
        {
            var addParenthesis = this._left.Parents.Any();
            var leftStr = addParenthesis ? $"({this._left})" : $"{this._left}";

            addParenthesis = this._right.Parents.Any();
            var rightStr = addParenthesis ? $"({this._right})" : $"{this._right}";

            return $"{leftStr} * {rightStr}";
        }
    }
}