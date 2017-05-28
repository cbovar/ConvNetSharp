using System;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Ops
{
    [DebuggerDisplay("{Name}")]
    public class Variable<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Variable(Volume<T> v, string name)
        {
            this.Name = name;
            this.V = v;
        }

        public Volume<T> V { get; set; }

        public string Name { get; set; }

        public override string Representation => this.Name;

        public override void Backward()
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.V?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Forward(Session<T> session)
        {
            return this.V;
        }

        public override string ToString()
        {
            return this.Name;
        }
    }
}