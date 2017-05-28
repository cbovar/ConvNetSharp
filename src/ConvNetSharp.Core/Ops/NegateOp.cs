using System;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Ops
{
    /// <summary>
    ///     y = -x
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class NegateOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> _result;

        public override void Backward()
        {
            if (this.Parents[0].Derivate == null)
            {
                this.Parents[0].Derivate = -this.Derivate;
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

        public override Volume<T> Forward(Session<T> session)
        {
            var y = this.Parents[0].Forward(session);

            if (this._result == null || !Equals(this._result.Shape, y.Shape))
            {
                this._result = BuilderInstance<T>.Volume.SameAs(y.Shape);
            }

            y.DoNegate(this._result);

            return this._result;
        }

        public override string ToString()
        {
            var addParenthesis = this.Parents[0].Parents.Any();
            if (addParenthesis)
            {
                return $"(-{this.Parents[0]})";
            }

            return $"-{this.Parents[0]}";
        }
    }
}