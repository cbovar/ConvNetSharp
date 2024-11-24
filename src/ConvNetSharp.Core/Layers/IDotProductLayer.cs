using System.Numerics;

namespace ConvNetSharp.Core.Layers;

public interface IDotProductLayer<T> where T : INumber<T>, IMinMaxValue<T>
{
    T BiasPref { get; set; }
}