using System;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = -x
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class NegateOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> _result;
        private readonly Op<T> _x;

        public NegateOp(Op<T> x)
        {
            this._x = x;
            AddParent(x);
        }

        public override void Differentiate()
        {
            if (this._x.Derivate == null)
            {
                this._x.Derivate = -this.Derivate;
            }
        }

        public override string Representation => "Neg";

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

            var y = this._x.Evaluate(session);

            if (this._result == null || !Equals(this._result.Shape, y.Shape))
            {
                this._result?.Dispose();
                this._result = BuilderInstance<T>.Volume.SameAs(y.Shape);
            }

            y.DoNegate(this._result);

            return this._result;
        }

        public override string ToString()
        {
            var addParenthesis = this._x.Parents.Any();
            if (addParenthesis)
            {
                return $"(-{this._x})";
            }

            return $"-{this._x}";
        }
    }
}