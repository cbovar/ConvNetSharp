using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public interface IPersistable<T> : INamedOp<T> where T : struct, IEquatable<T>, IFormattable
    {
        Volume<T> Result { get; set; }
    }
}