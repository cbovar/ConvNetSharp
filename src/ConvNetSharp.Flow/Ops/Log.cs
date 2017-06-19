using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Log<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;

        public Log(Op<T> x)
        {
            this._x = x;
            this.AddParent(x);
        }

        public override string Representation => "Log";

        public override void Differentiate()
        {
            this._x.RegisterDerivate(this.Derivate / this._x);
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

            x.DoLog(this.Result);
            return this.Result;
        }
    }
}