using System;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Element wise division
    ///     y = left / right
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class DivOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _left;
        private readonly Op<T> _right;

        public DivOp(Op<T> left, Op<T> right)
        {
            this._left = left;
            this._right = right;

            AddParent(left);
            AddParent(right);
        }

        public override string Representation => "/";

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
            if (this.LastComputeStep == session.Step)
            {
                return this.Result;
            }
            this.LastComputeStep = session.Step;

            var left = this._left.Evaluate(session);
            var right = this._right.Evaluate(session);

            if (!Object.Equals(left.Shape, right.Shape))
            {
                throw new ArgumentException("Both volume should have the same shape.");
            }

            if (this.Result == null || !Object.Equals(this.Result.Shape, left.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(left.Shape);
            }

            left.DoDivide(right, this.Result);

            return this.Result;
        }

        public override string ToString()
        {
            var addParenthesis = Enumerable.Any(this._left.Parents);
            var leftStr = addParenthesis ? $"({this._left})" : $"{this._left}";

            addParenthesis = Enumerable.Any(this._right.Parents);
            var rightStr = addParenthesis ? $"({this._right})" : $"{this._right}";

            return $"{leftStr} / {rightStr}";
        }
    }
}