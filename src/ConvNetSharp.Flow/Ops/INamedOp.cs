using System;

namespace ConvNetSharp.Flow.Ops
{
    public interface INamedOp<T> where T : struct, IEquatable<T>, IFormattable
    {
        string Name { get; set; }
    }
}