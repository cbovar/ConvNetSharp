using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Implements LeakyReLU nonlinearity elementwise
    ///     x -> x > 0, x, otherwise 0.01x
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class LeakyRelu<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public LeakyRelu(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public LeakyRelu(ConvNetSharp<T> graph, Op<T> x) : base(graph)
        {
            AddParent(x);
        }

        public override string Representation => "LeakyRelu";

        public override void Differentiate()
        {
            var x = this.Parents[0];

            x.RegisterDerivate(this.Graph.LeakyReluGradient(this, this.Derivate));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, x.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoLeakyRelu(this.Result);
            return this.Result;
        }

        public override string ToString()
        {
            return $"LeakyRelu({this.Parents[0]})";
        }
    }
}