using System;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    [DebuggerDisplay("{Name}")]
    public class PlaceHolder<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public PlaceHolder(string name)
        {
            this.Name = name;
        }

        public Volume<T> V { get; set; }

        public string Name { get; }

        public override void Differentiate()
        {
        }

        public override string Representation => this.Name;

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.V?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            return this.V;
        }

        public override string ToString()
        {
            return this.Name;
        }
    }


}