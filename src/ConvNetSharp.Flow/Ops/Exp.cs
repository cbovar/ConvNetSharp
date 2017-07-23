using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Exp<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Exp(Dictionary<string, object> data)
        {
        }

        public Exp(Op<T> x)
        {
            AddParent(x);
        }

        public override string Representation => "Exp";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(this.Derivate * new Exp<T>(this.Parents[0]));
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

            x.DoExp(this.Result);
            return this.Result;
        }

        public override string ToString()
        {
            return $"Exp({this.Parents[0]})";
        }
    }
}