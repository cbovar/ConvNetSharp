using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers
{
    public interface ILastLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        Op<T> Cost { get; }
    }
}