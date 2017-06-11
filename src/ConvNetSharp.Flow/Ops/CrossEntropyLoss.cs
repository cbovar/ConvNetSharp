using System;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class CrossEntropyLoss<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;
        private readonly Op<T> _y;
        private Volume<T> _pj;
        private Volume<T> _logpj;
        private Volume<T> _temp;

        public CrossEntropyLoss(Op<T> x, Op<T> y)
        {
            this._x = x;
            this._y = y;
            AddParent(x);
            AddParent(y);
        }

        public override string Representation => "CrossEntropyLoss";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(this - this.Derivate);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this.LastComputeStep == session.Step)
            {
                return this._pj;
            }
            this.LastComputeStep = session.Step;

            // Loss = -Sum(yj * log pj)

            var x = this._x.Evaluate(session);
            var y = this._y.Evaluate(session);

            if (this._pj == null || !Equals(this._pj.Shape, x.Shape))
            {
                this._pj?.Dispose();
                this._pj = BuilderInstance<T>.Volume.SameAs(x.Shape);

                this._logpj?.Dispose();
                this._logpj = BuilderInstance<T>.Volume.SameAs(x.Shape);

                this._temp?.Dispose();
                this._temp = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoSoftMax(this._pj);
            this._pj.DoLog(this._logpj);

            y.DoMultiply(this._logpj, this._temp);


            return this._pj;
        }
    }
}