using System;
using ConvNetSharp.Core;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function#945918
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class SoftMaxCrossEntropy<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _pj;
        private readonly Op<T> _x;
        private readonly Op<T> _y;
        private Volume<T> _logpj;
        private Volume<T> _temp;

        public SoftMaxCrossEntropy(Op<T> x, Op<T> y)
        {
            this._x = x;
            this._y = y;
            AddParent(x);
            AddParent(y);

            this._pj = ConvNetSharp<T>.Instance.Softmax(this._x); // pj = softmax(oj) = exp(oj)/Sum(exp(ok))

            // AddParent(epsilon);

            this.Result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 1));
        }

        public override string Representation => "SoftMaxCrossEntropy";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(this._pj - this._y); // dL/do = p - y
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            //// Loss = -Sum(yj * log pj)
            //var x = this._x.Evaluate(session);
            var y = this._y.Evaluate(session);
            var outputActivation = this._pj.Evaluate(session);

            //if (this._logpj == null || !Equals(this._logpj.Shape, x.Shape))
            //{
            //    this._logpj?.Dispose();
            //    this._logpj = BuilderInstance<T>.Volume.SameAs(x.Shape);

            //    this._temp?.Dispose();
            //    this._temp = BuilderInstance<T>.Volume.SameAs(x.Shape);

            //    var outputShape = new Shape(x.Shape.GetDimension(-1));
            //    this.Result?.Dispose();
            //    this.Result = BuilderInstance<T>.Volume.SameAs(outputShape);
            //}

            //var pj = this._pj.Evaluate(session);
            //pj.DoLog(this._logpj);

            //y.DoMultiply(this._logpj, this._temp);

            //this._temp.DoSum(this.Result);

            //this.Result.DoNegate(this.Result);

            //loss is the class negative log likelihood

            var loss = Ops<T>.Zero;
            for (var n = 0; n < y.Shape.GetDimension(3); n++)
            {
                for (var d = 0; d < y.Shape.GetDimension(2); d++)
                {
                    for (var h = 0; h < y.Shape.GetDimension(1); h++)
                    {
                        for (var w = 0; w < y.Shape.GetDimension(0); w++)
                        {
                            var expected = y.Get(w, h, d, n);
                            var actual = outputActivation.Get(w, h, d, n);
                            if (Ops<T>.Zero.Equals(actual))
                            {
                                actual = Ops<T>.Epsilon;
                            }
                            var current = Ops<T>.Multiply(expected, Ops<T>.Log(actual));

                            loss = Ops<T>.Add(loss, current);
                        }
                    }
                }
            }

            var batchSize = outputActivation.Shape.GetDimension(3);
            loss = Ops<T>.Divide(Ops<T>.Negate(loss), Ops<T>.Cast(batchSize));
            this.Result.Set(0, loss);

            return this.Result;
        }
    }
}