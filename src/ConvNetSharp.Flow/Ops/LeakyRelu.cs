using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Implements LeakyReLU non-linearity element-wise
    ///     x -> x > 0, x, otherwise alpha * x
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class LeakyRelu<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public LeakyRelu(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this.Alpha = (T)Convert.ChangeType(data["Alpha"], typeof(T));
        }

        public LeakyRelu(ConvNetSharp<T> graph, Op<T> x, T alpha) : base(graph)
        {
            this.Alpha = alpha;
            this.AddParent(x);
        }

        public T Alpha { get; set; }

        public override string Representation => "LeakyRelu";

        public override void Differentiate()
        {
            var x = this.Parents[0];

            x.RegisterDerivate(this.Graph.LeakyReluGradient(this, this.Derivate, this.Alpha));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, x.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.LeakyRelu(this.Alpha, this.Result);

            return base.Evaluate(session);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["Alpha"] = this.Alpha;
            return data;
        }

        public override string ToString()
        {
            return $"LeakyRelu({this.Parents[0]})";
        }
    }
}