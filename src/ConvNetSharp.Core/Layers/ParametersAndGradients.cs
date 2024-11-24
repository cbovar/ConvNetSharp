using System.Numerics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers;

public class ParametersAndGradients<T> where T : INumber<T>, IMinMaxValue<T>
{
    public Volume<T> Volume { get; set; }

    public Volume<T> Gradient { get; set; }
}