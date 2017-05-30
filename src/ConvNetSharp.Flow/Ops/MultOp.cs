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
    public class MultOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _left;
        private readonly Op<T> _right;
        private Volume<T> _result;

        public MultOp(Op<T> left, Op<T> right)
        {
            this._left = left;
            this._right = right;

            AddParent(left);
            AddParent(right);
        }

        public override string Representation => "*";

        public override void Differentiate()
        {
            if (this._left.Derivate == null)
            {
                this._left.Derivate = this.Derivate * this._right;
            }
            else
            {
                this._left.Derivate += this.Derivate * this._right;
            }

            if (this._right.Derivate == null)
            {
                this._right.Derivate = this.Derivate * this._left;
            }
            else
            {
                this._right.Derivate += this.Derivate * this._left;
            }
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
            if (this.LastComputeStep == session.Step)
            {
                return this._result;
            }
            this.LastComputeStep = session.Step;

            var left = this._left.Evaluate(session);
            var right = this._right.Evaluate(session);

            if (!Equals(left.Shape, right.Shape))
            {
                throw new ArgumentException("Both volume should have the same shape.");
            }

            if (this._result == null || !Equals(this._result.Shape, left.Shape))
            {
                this._result?.Dispose();
                this._result = BuilderInstance<T>.Volume.SameAs(left.Shape);
            }

            left.DoMultiply(right, this._result);

            return this._result;
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