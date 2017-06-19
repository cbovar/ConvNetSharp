using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Exp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;

        public Exp(Op<T> x)
        {
            this._x = x;
            AddParent(x);
        }

        public override string Representation => "Exp";

        public override void Differentiate()
        {
            this._x.RegisterDerivate(this.Derivate * new Exp<T>(this._x));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var x = this._x.Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, x.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoExp(this.Result);
            return this.Result;
        }
    }
}