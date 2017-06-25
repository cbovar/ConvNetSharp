using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Max<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Max(Dictionary<string, object> data)
        {
        }

        public Max(Op<T> x)
        {
            AddParent(x);
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
                return this.Result;
            }
            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);
            var reshape = x.ReShape(-1, x.Shape.GetDimension(-1));
            var targetShape = new Shape(reshape.Shape);
            targetShape.SetDimension(0, 1);

            if (this.Result == null || !Equals(this.Result.Shape, targetShape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(targetShape);
            }

            reshape.DoReduce(this.Result, TensorReduceOp.Max);

            return this.Result;
        }

        public override string ToString()
        {
            return $"Max({this.Parents[0]})";
        }
    }
}