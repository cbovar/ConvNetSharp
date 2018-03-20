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
                return base.Evaluate(session);
            }
            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, x.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoSqrt(this.Result);
            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"Sqrt({this.Parents[0]})";
        }
    }
}