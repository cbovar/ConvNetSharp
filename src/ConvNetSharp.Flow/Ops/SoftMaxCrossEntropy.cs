using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function#945918
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class SoftmaxCrossEntropy<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public SoftmaxCrossEntropy(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this.Result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 1));
        }

        public SoftmaxCrossEntropy(ConvNetSharp<T> graph, Op<T> softmax, Op<T> y) : base(graph)
        {
            this.AddParent(softmax);
            this.AddParent(y);

            this.Result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 1));
        }

        public override string Representation => "SoftmaxCrossEntropy";

        public override void Differentiate()
        {
            // here we skip softmax op
            this.Parents[0].Parents[0].RegisterDerivate(this.Parents[0] - this.Parents[1]); // dL/do = p - y
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var y = this.Parents[1].Evaluate(session);
            var outputActivation = this.Parents[0].Evaluate(session);

            var loss = Ops<T>.Zero;
            for (var n = 0; n < y.Shape.Dimensions[3]; n++)
            {
                for (var d = 0; d < y.Shape.Dimensions[2]; d++)
                {
                    for (var h = 0; h < y.Shape.Dimensions[1]; h++)
                    {
                        for (var w = 0; w < y.Shape.Dimensions[0]; w++)
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

            var batchSize = outputActivation.Shape.Dimensions[3];
            loss = Ops<T>.Divide(Ops<T>.Negate(loss), Ops<T>.Cast(batchSize));
            this.Result.Set(0, loss);

            return base.Evaluate(session);
        }
    }
}