using System;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Ops
{
    /// <summary>
    ///     Element wise
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class MultOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> _result;

        public override void Backward()
        {
            if (this.Parents[0].Derivate == null)
            {
                this.Parents[0].Derivate = this.Derivate * this.Parents[1];
            }
            else
            {
                this.Parents[0].Derivate += this.Derivate * this.Parents[1];
            }

            if (this.Parents[1].Derivate == null)
            {
                this.Parents[1].Derivate = this.Derivate * this.Parents[0];
            }
            else
            {
                this.Parents[1].Derivate += this.Derivate * this.Parents[0];
            }
        }

        public override string Representation => "*";

        public override Volume<T> Forward(Session<T> session)
        {
            var left = this.Parents[0].Forward(session);
            var right = this.Parents[1].Forward(session);

            if (!Equals(left.Shape, right.Shape))
            {
                throw new ArgumentException("Both volume should have the same shape.");
            }

            if (this._result == null || !Equals(this._result.Shape, left.Shape))
            {
                this._result = BuilderInstance<T>.Volume.SameAs(left.Shape);
            }

            left.DoMultiply(right, this._result);

            return this._result;
        }

        public override string ToString()
        {
            var addParenthesis = this.Parents[0].Parents.Any();
            var leftStr = addParenthesis ? $"({this.Parents[0]})" : $"{this.Parents[0]}";

            addParenthesis = this.Parents[1].Parents.Any();
            var rightStr = addParenthesis ? $"({this.Parents[1]})" : $"{this.Parents[1]}";

            return $"{leftStr} * {rightStr}";
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this._result?.Dispose();
            }

            base.Dispose(disposing);
        }
    }
}