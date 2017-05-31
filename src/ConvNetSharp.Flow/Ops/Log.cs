using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Log<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;
        private Volume<T> _result;

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
            if (this.LastComputeStep == session.Step)
            {
                return this._result;
            }
            this.LastComputeStep = session.Step;

            var x = this._x.Evaluate(session);

            if (this._result == null || !Equals(this._result.Shape, x.Shape))
            {
                this._result?.Dispose();
                this._result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoLog(this._result);
            return this._result;
        }
    }
}