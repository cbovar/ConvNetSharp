using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers
{
    public abstract class LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Op<T> Op { get; set; }

        public virtual void AcceptParent(Op<T> parent)
        {
        }
    }
}