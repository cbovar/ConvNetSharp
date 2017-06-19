using System;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    /// y = C where C is a constant
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [DebuggerDisplay("{Name}")]
    public class Const<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Const(Volume<T> v, string name)
        {
            this.Name = name;
            this.Result = v;
        }

        public string Name { get; set; }

        public override void Differentiate()
        {
        }

        public override string Representation => this.Name;

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