using System;
using System.Collections.Generic;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Variable hold a named Volume that can change over time.
    /// </summary>
    [DebuggerDisplay("{" + nameof(Name) + "}")]
    public class Variable<T> : Op<T>, IPersistable<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Variable(ConvNetSharp<T> graph, Volume<T> v, string name, bool isLearnable = false) : base(graph)
        {
            this.Name = name;
            this.Result = v;
            this.IsLearnable = isLearnable;
        }

        public Variable(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this.Name = (string)data["Name"];
            this.IsLearnable = (string)data["IsLearnable"] == "True";
        }

        public override string Representation => this.Name;

        /// <summary>
        ///     If set to true optimizer will try to reduce cost by updating this variable
        /// </summary>
        public bool IsLearnable { get; }

        public string Name { get; set; }

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
            data["IsLearnable"] = this.IsLearnable;
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