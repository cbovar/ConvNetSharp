using System.Numerics;

namespace ConvNetSharp.Flow.Layers;

public class LeakyReluLayer<T>(T alpha) : LayerBase<T>
    where T : INumber<T>, IMinMaxValue<T>
{
    public T Alpha { get; set; } = alpha;

    public override void AcceptParent(LayerBase<T> parent)
    {
        base.AcceptParent(parent);
        this.Op = parent.Op.Graph.LeakyRelu(parent.Op, this.Alpha);
    }
}