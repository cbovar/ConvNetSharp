using System;
using System.Collections.Generic;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     A PlaceHolder is an entry point in the computation graph. Its value is set at every run of the graph using the
    ///     provided dictionary (See ConvNetSharp.Flow.Session.UpdatePlaceHolder method)
    /// </summary>
    [DebuggerDisplay("{" + nameof(Name) + "}")]
    public class PlaceHolder<T> : Op<T>, INamedOp<T> where T : struct, IEquatable<T>, IFormattable
    {
        public PlaceHolder(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this.Name = (string) data["Name"];
        }

        public PlaceHolder(ConvNetSharp<T> graph, string name) : base(graph)
        {
            this.Name = name;
        }

        public override string Representation => this.Name;

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

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["Name"] = this.Name;
            return data;
        }

        public void SetValue(Volume<T> value)
        {
            this.Result = value;
            this.SetDirty();
        }

        public override string ToString()
        {
            return this.Name;
        }
    }
}