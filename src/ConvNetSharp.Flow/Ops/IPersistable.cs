using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public interface IPersistable<T> where T : struct, IEquatable<T>, IFormattable
    {
        string Name { get; }

        Volume<T> Result { get; set; }
    }
}