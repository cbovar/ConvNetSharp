using System.Collections.Generic;
using System.Numerics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers;

public abstract class LastLayerBase<T> : LayerBase<T>, ILastLayer<T> where T : INumber<T>, IMinMaxValue<T>
{
    protected LastLayerBase()
    {
    }

    protected LastLayerBase(Dictionary<string, object> data) : base(data)
    {
    }

    public abstract void Backward(Volume<T> y, out T loss);
}