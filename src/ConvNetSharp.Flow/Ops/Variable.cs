using System;
using System.Collections.Generic;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    [DebuggerDisplay("{Name}")]
    public class Variable<T> : Op<T>, IPersistable<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Variable(Volume<T> v, string name)
        {
            this.Name = name;
            this.Result = v;
        }

        public Variable(Dictionary<string, object> data)
        {
            this.Name = (string) data["Name"];
        }

        public string Name { get; set; }

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