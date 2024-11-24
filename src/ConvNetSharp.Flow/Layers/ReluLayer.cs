using System.Numerics;

namespace ConvNetSharp.Flow.Layers;

public class ReluLayer<T> : LayerBase<T> where T : INumber<T>, IMinMaxValue<T>
{
    public override void AcceptParent(LayerBase<T> parent)
    {
        base.AcceptParent(parent);
        this.Op = parent.Op.Graph.Relu(parent.Op);
    }
}