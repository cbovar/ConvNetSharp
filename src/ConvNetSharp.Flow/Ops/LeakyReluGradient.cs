using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Implements LeakyReLU gradient
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class LeakyReluGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public LeakyReluGradient(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this.Alpha = (T)Convert.ChangeType(data["Alpha"], typeof(T));
        }

        public LeakyReluGradient(ConvNetSharp<T> graph, Op<T> y, Op<T> derivate, T alpha) : base(graph)
        {
            this.Alpha = alpha;
            this.AddParent(y);
            this.AddParent(derivate);
        }

        public T Alpha { get; set; }

        public override string Representation => "LeakyReluGradient";

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var y = this.Parents[0].Evaluate(session);
            var derivate = this.Parents[1].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, y.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(y.Shape);
            }

            y.LeakyReluGradient(derivate, this.Result, this.Alpha);
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
            return $"LeakyReluGradient({this.Parents[0]})";
        }
    }
}