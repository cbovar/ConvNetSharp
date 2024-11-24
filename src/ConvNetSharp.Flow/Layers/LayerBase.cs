using System.Numerics;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers;

public abstract class LayerBase<T> where T : INumber<T>, IMinMaxValue<T>
{
    public Op<T> Op { get; set; }

    public int Id { get; private set; }

    public virtual void AcceptParent(LayerBase<T> parent)
    {
        this.Id = parent.Id + 1;
    }
}