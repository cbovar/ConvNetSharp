using System.Numerics;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers;

public interface ILastLayer<T> where T : INumber<T>, IMinMaxValue<T>
{
    Op<T> Cost { get; }
}