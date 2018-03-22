using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    /// Computes u^v
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Power<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Power(Dictionary<string, object> data)
        {
        }

        public Power(Op<T> u, Op<T> v)
        {
            AddParent(u);
            AddParent(v);
        }

        public override string Representation => "Pow";

        public override void Differentiate()
        {
            var u = this.Parents[0];
            var v = this.Parents[1];

            // d(u^v)/d(u) = v.u^(v-1)
            u.RegisterDerivate(this.Derivate * v * (u ^ (v - new Const<T>(ConvNetSharp<T>.One, "one"))));

            // d(u^v)/d(v) = u^v.log(u)
            v.RegisterDerivate(this.Derivate * (u ^ v) * new Log<T>(u));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var u = this.Parents[0].Evaluate(session);
            var v = this.Parents[1].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, u.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(u.Shape);
            }

            u.DoPower(v, this.Result);
            return this.Result;
        }

        public override string ToString()
        {
            return $"Pow({this.Parents[0]}, {this.Parents[1]})";
        }
    }
}