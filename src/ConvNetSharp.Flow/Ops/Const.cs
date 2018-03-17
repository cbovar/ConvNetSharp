using System;
using System.Collections.Generic;
using System.Diagnostics;
using ConvNetSharp.Volume;
using System.Linq;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = C where C is a constant
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [DebuggerDisplay("{Name}")]
    public class Const<T> : Op<T>, IPersistable<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly T _x;
        private int _lastBatchSize;
        private Shape _tempShape;

        public Const(Dictionary<string, object> data)
        {
            this.Name = (string)data["Name"];
            this._x = (T) data["x"];

            if (data.ContainsKey("dim0"))
            {
                var dim0 = int.Parse((string)data["dim0"]);
                var dim1 = int.Parse((string)data["dim1"]);
                var dim2 = int.Parse((string)data["dim2"]);
                var dim3 = int.Parse((string)data["dim3"]);

                this.OutputShape = new Shape(dim0, dim1, dim2, dim3);
            }
        }

        public Const(Volume<T> v, string name)
        {
            this.Name = name;
            this.Result = v;
            this.OutputShape = this.Result.Shape;
        }

        public Const(T x, Op<T> shape, string name)
        {
            this.Name = name;
            this._x = x;
            AddParent(shape);
        }

        public string Name { get; set; }

        public override string Representation => this.Name;

        public override void Differentiate()
        {
        }

        public Shape OutputShape { get; }

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
            if (this.Result == null)
            {
                if (this.OutputShape != null)
                {
                    this.Result = BuilderInstance<T>.Volume.From(new T[this.OutputShape.TotalLength].Populate(this._x), this.OutputShape);
                }
                else
                {
                    if (this._tempShape == null || session.BatchSize != this._lastBatchSize)
                    {
                        var shape = this.Parents[0].Evaluate(session);
                        var s = new[] {shape.Get(0), shape.Get(1), shape.Get(2), shape.Get(3)};
                        var t = s.Select(o => Convert.ToInt32(o)).ToArray();
                        this._tempShape = new Shape(t);
                        this._lastBatchSize = session.BatchSize;

                        this.Result = BuilderInstance<T>.Volume.From(new T[this._tempShape.TotalLength].Populate(this._x), this._tempShape);
                    }
                }
            }


            return this.Result;
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["Name"] = this.Name;
            data["x"] = this._x;

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
            return this.Name;
        }
    }
}