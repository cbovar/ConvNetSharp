using System;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Reshape<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Shape _outputShape;
        private readonly Op<T> _shape;
        private readonly Op<T> _x;
        private Shape _tempShape;
        private int lastBatchSize;

        public Reshape(Op<T> x, Shape shape)
        {
            this._x = x;
            AddParent(x);

            this._outputShape = shape;
        }

        public Reshape(Op<T> x, Op<T> shape)
        {
            this._x = x;
            this._shape = shape;
            AddParent(x);
            AddParent(shape);
        }

        public override string Representation => $"Reshape ({this._outputShape.PrettyPrint(",")})";

        public override void Differentiate()
        {
            this._x.RegisterDerivate(ConvNetSharp<T>.Instance.Reshape(this.Derivate, ConvNetSharp<T>.Instance.Shape(this._x)));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var y = this._x.Evaluate(session);

            if (this._outputShape != null)
            {
                this.Result = y.ReShape(this._outputShape);
            }
            else
            {
                if (this._tempShape == null || session.BatchSize != this.lastBatchSize)
                {
                    var shape = this._shape.Evaluate(session);
                    var s = new[] { shape.Get(0), shape.Get(1), shape.Get(2), shape.Get(3) };
                    var t = s.Select(o => Convert.ToInt32(o)).ToArray();
                    this._tempShape = new Shape(t);
                    this.lastBatchSize = session.BatchSize;
                }

                this.Result = y.ReShape(this._tempShape);
            }

            return this.Result;
        }
    }
}