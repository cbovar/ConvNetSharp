using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Transpose<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Transpose(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public Transpose(ConvNetSharp<T> graph, Op<T> x) : base(graph)
        {
            this.AddParent(x);
        }

        public override string Representation => "Transpose";

        public override void Differentiate()
        {
            // TODO
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);

            var expectedShape = new Shape(x.Shape.Dimensions[1], x.Shape.Dimensions[0], 1, x.Shape.Dimensions[3]);

            if (this.Result == null || !Equals(this.Result.Shape, expectedShape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(expectedShape);
            }

            x.Transpose(this.Result);
            return base.Evaluate(session);
        }

        public override string ToString()
        {
            if (this.Parents[0].Parents.Count <= 1)
            {
                return $"{this.Parents[0]}ᵀ";
            }

            return $"({this.Parents[0]})ᵀ";
        }
    }
}