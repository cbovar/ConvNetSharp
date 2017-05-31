using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = a + b
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class AddOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _left;
        private readonly Op<T> _right;
        private Volume<T> _result;

        public AddOp(Op<T> left, Op<T> right)
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
                this._result?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this.LastComputeStep == session.Step) return this._result;
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

            left.DoAdd(right, this._result);

            return this._result;
        }

        public override string ToString()
        {
            return $"{this._left} + {this._right}";
        }
    }
}