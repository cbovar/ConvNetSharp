using System;
using ConvNetSharp.Volume;
using ConvNetSharp.Core;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///  https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function#945918
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class SoftMaxCrossEntropy<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;
        private readonly Op<T> _y;
        private Volume<T> _logpj;
        private Volume<T> _temp;
        private readonly Op<T> _pj;

        public SoftMaxCrossEntropy(Op<T> x, Op<T> y)
        {
            this._x = x;
            this._y = y;
            AddParent(x);
            AddParent(y);

            var epsilon = ConvNetSharp<T>.Instance.Const(Ops<T>.Epsilon, "epsilon");
            this._pj = ConvNetSharp<T>.Instance.Softmax(this._x) + epsilon; // pj = softmax(oj) = exp(oj)/Sum(exp(ok))

            AddParent(epsilon);
        }

        public override string Representation => "SoftMaxCrossEntropy";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(this._pj - this._y); // dL/do = p - y
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this.LastComputeStep == session.Step)
            {
                return this.Result;
            }
            this.LastComputeStep = session.Step;

            // Loss = -Sum(yj * log pj)

            var x = this._x.Evaluate(session);
            var y = this._y.Evaluate(session);

            if (this._logpj == null || !Equals(this._logpj.Shape, x.Shape))
            {
                this._logpj?.Dispose();
                this._logpj = BuilderInstance<T>.Volume.SameAs(x.Shape);

                this._temp?.Dispose();
                this._temp = BuilderInstance<T>.Volume.SameAs(x.Shape);

                var outputShape = new Shape(x.Shape.GetDimension(-1));
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(outputShape);
            }

            var pj = this._pj.Evaluate(session);
            pj.DoLog(this._logpj);

            y.DoMultiply(this._logpj, this._temp);

            this._temp.DoSum(this.Result);

            this.Result.DoNegate(this.Result);

#if DEBUG
            var inputs = Result.ToArray();
            foreach (var i in inputs)
            {
                if (Ops<T>.IsInvalid(i))
                    throw new ArgumentException("Invalid input!");
            }
#endif

            return this.Result;
        }
    }
}