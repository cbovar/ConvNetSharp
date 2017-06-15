using System;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    [DebuggerDisplay("{Name}")]
    public class Variable<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Variable(Volume<T> v, string name)
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

        public override string ToString()
        {
            return this.Name;
        }
    }
}