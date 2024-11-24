using System.Numerics;

namespace ConvNetSharp.Flow.Layers;

public class InputLayer<T> : LayerBase<T> where T : INumber<T>, IMinMaxValue<T>
{
    public InputLayer()
    {
        var cns = new ConvNetSharp<T>();
        this.Op = cns.PlaceHolder("input");
    }
}