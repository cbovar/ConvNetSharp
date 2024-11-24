using System.Numerics;

namespace ConvNetSharp.Flow.Layers;

public class LeakyReluLayer<T> : LayerBase<T> where T : INumber<T>, IMinMaxValue<T>
{
    public LeakyReluLayer(T alpha)
    {
        this.Alpha = alpha;
    }

    public T Alpha { get; set; }

    public override void AcceptParent(LayerBase<T> parent)
    {
        base.AcceptParent(parent);
        this.Op = parent.Op.Graph.LeakyRelu(parent.Op, this.Alpha);
    }
}