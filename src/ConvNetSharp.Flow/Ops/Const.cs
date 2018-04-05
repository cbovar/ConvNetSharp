using System;
using System.Collections.Generic;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Const hold a Volume that will not change over time.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [DebuggerDisplay("{Name}")]
    public class Const<T> : Op<T>, IPersistable<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly T _x;

        public Const(Dictionary<string, object> data)
        {
            this.Name = (string)data["Name"];
            this._x = (T)Convert.ChangeType(data["x"], typeof(T));

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

        public Const(T x, string name)
        {
            this.Name = name;
            this._x = x;
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
                    this.Result = BuilderInstance<T>.Volume.From(new[] { this._x }, new Shape(1, 1, 1, 1));
                }
            }

            return base.Evaluate(session);
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