using System.Numerics;

namespace ConvNetSharp.Flow.Layers;

public class TanhLayer<T> : LayerBase<T> where T : INumber<T>, IMinMaxValue<T>
{
    public override void AcceptParent(LayerBase<T> parent)
    {
        base.AcceptParent(parent);
        this.Op = parent.Op.Graph.Tanh(parent.Op);
    }
}