using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Sum<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private int _lastBatchSize;
        private Shape _tempShape;

        public Sum(Op<T> x, Shape shape)
        {
            AddParent(x);

            this.OutputShape = shape;
        }

        public Sum(Op<T> x, Op<T> shape)
        {
            AddParent(x);
            AddParent(shape);
        }

        public Sum(Dictionary<string, object> data)
        {
            if (data.ContainsKey("dim0"))
            {
                var dim0 = int.Parse((string) data["dim0"]);
                var dim1 = int.Parse((string) data["dim1"]);
                var dim2 = int.Parse((string) data["dim2"]);
                var dim3 = int.Parse((string) data["dim3"]);

                this.OutputShape = new Shape(dim0, dim1, dim2, dim3);
            }
        }

        public Shape OutputShape { get; }

        public override string Representation => $"Sum ({this.OutputShape?.PrettyPrint(",")})";

        public override void Differentiate()
        {
            // Repeat gradient so that it matches input shape
            this.Parents[0].RegisterDerivate(ConvNetSharp<T>.Instance.Tile(this.Derivate,
                ConvNetSharp<T>.Instance.Shape(this.Parents[0]) / ConvNetSharp<T>.Instance.Shape(this.Derivate)));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);
            var isScalar = x.Shape.TotalLength == 1;

            if (this.OutputShape != null)
            {
                if (this.Result == null)
                {
                    this.Result = BuilderInstance<T>.Volume.SameAs(this.OutputShape);
                }
            }
            else
            {
                if (this.Result == null || session.BatchSize != this._lastBatchSize)
                {
                    var shape = this.Parents[1].Evaluate(session);
                    var s = new[] { isScalar ? ConvNetSharp<T>.One : shape.Get(0), isScalar ? ConvNetSharp<T>.One : shape.Get(1), isScalar ? ConvNetSharp<T>.One : shape.Get(2), shape.Get(3)};
                    var t = s.Select(o => Convert.ToInt32(o)).ToArray();
                    this._tempShape = new Shape(t);
                    this._lastBatchSize = session.BatchSize;

                    this.Result = BuilderInstance<T>.Volume.SameAs(this._tempShape);
                }
            }

            this.Result.Clear();

            if (isScalar)
            {
                var shape = this.OutputShape != null ? this.OutputShape.ToVolume<T>() : this.Parents[1].Evaluate(session);
                x.DoTile(shape, this.Result);
            }
            else
            {
                x.DoReduce(this.Result, TensorReduceOp.Add);
            }

            return base.Evaluate(session);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();

            if (this.OutputShape != null)
            {
                data["dim0"] = this.OutputShape.GetDimension(0);
                data["dim1"] = this.OutputShape.GetDimension(1);
                data["dim2"] = this.OutputShape.GetDimension(2);
                data["dim3"] = this.OutputShape.GetDimension(3);
            }

            return data;
        }

        public override string ToString()
        {
            return $"sum({this.Parents[0]})";
        }
    }
}