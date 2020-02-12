using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Computes square root of x element-wise
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Sqrt<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Sqrt(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public Sqrt(ConvNetSharp<T> graph, Op<T> x) : base(graph)
        {
            this.AddParent(x);
        }

        public override string Representation => "Sqrt";

        public override void Differentiate()
        {
            var u = this.Parents[0];

            // d(sqrt(u))/du = 1 / (2*sqrt(u))
            u.RegisterDerivate(this.Derivate / (this.Graph.Const((T)Convert.ChangeType(2.0, typeof(T)), "two") * this.Graph.Sqrt(u)));
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

            x.Sqrt(this.Result);
            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"Sqrt({this.Parents[0]})";
        }
    }
}