using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    /// Computes square root of x element-wise
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Sqrt<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Sqrt(Dictionary<string, object> data)
        {
        }

        public Sqrt(Op<T> x)
        {
            AddParent(x);
        }

        public override string Representation => "Sqrt";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(this.Derivate / (new Const<T>((T)Convert.ChangeType(2.0, typeof(T)), "two") * new Sqrt<T>(this.Parents[0])));
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

            x.DoSqrt(this.Result);
            return this.Result;
        }

        public override string ToString()
        {
            return $"Sqrt({this.Parents[0]})";
        }
    }


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
            this.Parents[0].RegisterDerivate(this.Derivate * this.Parents[1] * new Power<T>(this.Parents[0], this.Parents[1] - new Const<T>(ConvNetSharp<T>.One, "one")));
            this.Parents[1].RegisterDerivate(this.Derivate * new Power<T>(this.Parents[0], this.Parents[1]) * new Log<T>(this.Parents[0]));
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