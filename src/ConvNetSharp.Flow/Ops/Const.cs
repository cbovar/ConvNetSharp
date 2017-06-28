using System;
using System.Collections.Generic;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = C where C is a constant
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [DebuggerDisplay("{Name}")]
    public class Const<T> : Op<T>, IPersistable<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Const(Dictionary<string, object> data)
        {
            this.Name = (string)data["Name"];
        }

        public Const(Volume<T> v, string name)
        {
            this.Name = name;
            this.Result = v;
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