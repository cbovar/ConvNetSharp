using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Core.Layers
{
    public interface ILastLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        Op<T> Cost { get; set; }
    }
}