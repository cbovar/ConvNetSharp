using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = -x
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Negate<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Negate(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public Negate(ConvNetSharp<T> graph, Op<T> x) : base(graph)
        {
            this.AddParent(x);
        }

        public override string Representation => "Neg";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(-this.Derivate);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Result?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var y = this.Parents[0].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, y.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(y.Shape);
            }

            y.Negate(this.Result);

            return base.Evaluate(session);
        }

        public override string ToString()
        {
            var addParenthesis = this.Parents[0].Parents.Any();
            if (addParenthesis)
            {
                return $"(-{this.Parents[0]})";
            }

            return $"-{this.Parents[0]}";
        }
    }
}