using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Max<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Max(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public Max(ConvNetSharp<T> graph, Op<T> x) : base(graph)
        {
            this.AddParent(x);
        }

        public override string Representation => "Max";

        public override void Differentiate()
        {
            throw new NotImplementedException();
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

            var x = this.Parents[0].Evaluate(session);
            var reshape = x.ReShape(-1, x.Shape.Dimensions[3]);
            var targetShape = new Shape(reshape.Shape);
            targetShape.SetDimension(0, 1);

            if (this.Result == null || !Equals(this.Result.Shape, targetShape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(targetShape);
            }

            reshape.Reduce(TensorReduceOp.Max, this.Result);

            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"Max({this.Parents[0]})";
        }
    }
}