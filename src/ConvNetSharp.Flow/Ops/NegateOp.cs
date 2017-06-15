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
        private readonly Op<T> _x;

        public NegateOp(Op<T> x)
        {
            this._x = x;
            AddParent(x);
        }

        public override void Differentiate()
        {
            this._x.RegisterDerivate(-this.Derivate);
        }

        public override string Representation => "Neg";

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
            if (this.LastComputeStep == session.Step) return this.Result;
            this.LastComputeStep = session.Step;

            var y = this._x.Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, y.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(y.Shape);
            }

            y.DoNegate(this.Result);

            return this.Result;
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