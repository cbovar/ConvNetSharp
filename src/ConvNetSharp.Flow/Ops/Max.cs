using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Max<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;
        private Volume<T> _result;

        public Max(Op<T> x)
        {
            this._x = x;
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
                this._result?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this.LastComputeStep == session.Step)
            {
                return this._result;
            }
            this.LastComputeStep = session.Step;

            var x = this._x.Evaluate(session);
            var reshape = x.ReShape(-1, x.Shape.GetDimension(-1));
            var targetShape = new Shape(reshape.Shape);
            targetShape.SetDimension(0,1);

            if (this._result == null || !Equals(this._result.Shape, targetShape))
            {
                this._result?.Dispose();
                this._result = BuilderInstance<T>.Volume.SameAs(targetShape);
            }

            reshape.DoReduce(this._result, TensorReduceOp.Max);

            return this._result;
        }

        public override string ToString()
        {
            return $"Max({this._x})";
        }
    }
}