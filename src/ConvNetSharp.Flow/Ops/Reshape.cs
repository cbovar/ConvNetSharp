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

        public Reshape(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
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

        public Reshape(ConvNetSharp<T> graph, Op<T> x, Shape shape) : base(graph)
        {
            this.AddParent(x);

            this.OutputShape = shape;
        }

        public Reshape(ConvNetSharp<T> graph, Op<T> x, Op<T> shape) : base(graph)
        {
            this.AddParent(x);
            this.AddParent(shape);
        }

        public Shape OutputShape { get; }

        public override string Representation => $"Reshape ({this.OutputShape?.PrettyPrint(",")})";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(this.Graph.Reshape(this.Derivate, this.Graph.Shape(this.Parents[0])));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
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
                    this._tempShape = new Shape(t[0], t[1], t[2], t[3]);
                    this._lastBatchSize = session.BatchSize;
                }

                this.Result = y.ReShape(this._tempShape);
            }

            return base.Evaluate(session);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();

            if (this.OutputShape != null)
            {
                data["dim0"] = this.OutputShape.Dimensions[0];
                data["dim1"] = this.OutputShape.Dimensions[1];
                data["dim2"] = this.OutputShape.Dimensions[2];
                data["dim3"] = this.OutputShape.Dimensions[3];
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