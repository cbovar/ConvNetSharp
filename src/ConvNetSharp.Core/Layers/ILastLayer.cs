using System.Numerics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers;

public interface ILastLayer<T> where T : INumber<T>, IMinMaxValue<T>
{
    void Backward(Volume<T> y, out T loss);
}