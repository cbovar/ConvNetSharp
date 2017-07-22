using System;
using System.Collections.Generic;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    [DebuggerDisplay("{Name}")]
    public class PlaceHolder<T> : Op<T>, INamedOp<T> where T : struct, IEquatable<T>, IFormattable
    {
        public PlaceHolder(Dictionary<string, object> data)
        {
            this.Name = (string)data["Name"];
        }

        public PlaceHolder(string name)
        {
            this.Name = name;
        }

        public string Name { get; }

        public override string Representation => this.Name;

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
            return this.Result;
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["Name"] = this.Name;
            return data;
        }

        public override string ToString()
        {
            return this.Name;
        }
    }
}