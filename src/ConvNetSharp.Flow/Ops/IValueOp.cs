using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public interface IValueOp<T> where T : struct, IEquatable<T>, IFormattable
    {
        void SetValue(Volume<T> value);
    }
}