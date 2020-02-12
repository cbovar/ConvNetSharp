using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Computes u^v
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Power<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Power(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public Power(ConvNetSharp<T> graph, Op<T> u, Op<T> v) : base(graph)
        {
            this.AddParent(u);
            this.AddParent(v);
        }

        public override string Representation => "Pow";

        public override void Differentiate()
        {
            var u = this.Parents[0];
            var v = this.Parents[1];

            // d(u^v)/d(u) = v.u^(v-1)
            u.RegisterDerivate(this.Derivate * v * (u ^ (v - this.Graph.Const(ConvNetSharp<T>.One, "one"))));

            // d(u^v)/d(v) = u^v.log(u)
            v.RegisterDerivate(this.Derivate * (u ^ v) * this.Graph.Log(u));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var u = this.Parents[0].Evaluate(session);
            var v = this.Parents[1].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, u.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(u.Shape);
            }

            u.Power(v, this.Result);
            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"Pow({this.Parents[0]}, {this.Parents[1]})";
        }
    }
}