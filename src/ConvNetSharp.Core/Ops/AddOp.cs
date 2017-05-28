using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Ops
{
    /// <summary>
    ///     y = a + b
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class AddOp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> _result;

        public override void Backward()
        {
            var derivate = this.Derivate;

            if (this.Parents[0].Derivate == null)
            {
                this.Parents[0].Derivate = derivate;
            }
            else
            {
                this.Parents[0].Derivate += derivate;
            }

            if (this.Parents[1].Derivate == null)
            {
                this.Parents[1].Derivate = derivate;
            }
            else
            {
                this.Parents[1].Derivate += derivate;
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

        public override string Representation => "+";

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

            left.DoAdd(right, this._result);

            return this._result;
        }

        public override string ToString()
        {
            return $"{this.Parents[0]} + {this.Parents[1]}";
        }
    }
}