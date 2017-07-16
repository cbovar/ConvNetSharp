using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Reshape<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private int _lastBatchSize;
        private Shape _tempShape;

        public Reshape(Dictionary<string, object> data)
        {
            if (data.ContainsKey("dim0"))
            {
                var dim0 = int.Parse((string)data["dim0"]);
                var dim1 = int.Parse((string)data["dim1"]);
                var dim2 = int.Parse((string)data["dim2"]);
                var dim3 = int.Parse((string)data["dim3"]);

                this.OutputShape = new Shape(dim0, dim1, dim2, dim3);
            }
        }

        public Reshape(Op<T> x, Shape shape)
        {
            AddParent(x);

            this.OutputShape = shape;
        }

        public Reshape(Op<T> x, Op<T> shape)
        {
            AddParent(x);
            AddParent(shape);
        }

        public Shape OutputShape { get; }

        public override string Representation => $"Reshape ({this.OutputShape?.PrettyPrint(",")})";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(ConvNetSharp<T>.Instance.Reshape(this.Derivate, ConvNetSharp<T>.Instance.Shape(this.Parents[0])));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var y = this.Parents[0].Evaluate(session);

            if (this.OutputShape != null)
            {
                this.Result = y.ReShape(this.OutputShape);
            }
            else
            {
                if (this._tempShape == null || session.BatchSize != this._lastBatchSize)
                {
                    var shape = this.Parents[1].Evaluate(session);
                    var s = new[] { shape.Get(0), shape.Get(1), shape.Get(2), shape.Get(3) };
                    var t = s.Select(o => Convert.ToInt32(o)).ToArray();
                    this._tempShape = new Shape(t);
                    this._lastBatchSize = session.BatchSize;
                }

                this.Result = y.ReShape(this._tempShape);
            }

            return this.Result;
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
            if (this.Parents[0] is Reshape<T>)
            {
                return this.Parents[0].ToString();
            }
            return $"reshape({this.Parents[0]})";
        }
    }
}