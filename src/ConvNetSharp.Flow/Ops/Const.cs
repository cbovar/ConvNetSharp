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
    [DebuggerDisplay("{" + nameof(Name) + "}")]
    public class Const<T> : Op<T>, IPersistable<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Const(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this.Name = (string)data["Name"];
            this.Value = (T)Convert.ChangeType(data["x"], typeof(T));

            if (data.ContainsKey("dim0"))
            {
                var dim0 = int.Parse((string)data["dim0"]);
                var dim1 = int.Parse((string)data["dim1"]);
                var dim2 = int.Parse((string)data["dim2"]);
                var dim3 = int.Parse((string)data["dim3"]);

                this.OutputShape = new Shape(dim0, dim1, dim2, dim3);
            }
        }

        public Const(ConvNetSharp<T> graph, Volume<T> v, string name) : base(graph)
        {
            this.Name = name;
            this.Result = v;
            this.OutputShape = this.Result.Shape;
        }

        public Const(ConvNetSharp<T> graph, T x, string name) : base(graph)
        {
            this.Name = name;
            this.Value = x;
        }

        public T Value { get; }

        public override string Representation => this.Name;

        public Shape OutputShape { get; }

        public string Name { get; set; }

        public override void Differentiate()
        {
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
            if (this.Result == null)
            {
                if (this.OutputShape != null)
                {
                    this.Result = BuilderInstance<T>.Volume.From(new T[this.OutputShape.TotalLength].Populate(this.Value), this.OutputShape);
                }
                else
                {
                    this.Result = BuilderInstance<T>.Volume.From(new[] { this.Value }, new Shape(1, 1, 1, 1));
                }
            }

            return base.Evaluate(session);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["Name"] = this.Name;
            data["x"] = this.Value;

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
            return this.Name;
        }
    }
}