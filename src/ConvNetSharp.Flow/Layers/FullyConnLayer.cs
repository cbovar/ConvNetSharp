using System.Numerics;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Layers;

public class FullyConnLayer<T>(int neuronCount) : LayerBase<T>
    where T : INumber<T>, IMinMaxValue<T>
{
    private Variable<T> _bias;
    private T _biasPref;
    private Dense<T> _conv;
    private bool _initialized;

    public T BiasPref
    {
        get => this._biasPref;
        set
        {
            this._biasPref = value;

            if (this._initialized)
            {
                this._bias.Result = new T[neuronCount].Populate(this.BiasPref);
            }
        }
    }

    public Op<T> Filter => this._conv.Filter;

    public override void AcceptParent(LayerBase<T> parent)
    {
        base.AcceptParent(parent);

        var cns = parent.Op.Graph;

        using (cns.Scope($"FullConnLayer_{this.Id}"))
        {
            this._bias = cns.Variable(BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, neuronCount, 1)), "Bias",
                true);
            this._conv = cns.Dense(cns.Reshape(parent.Op, new Shape(1, 1, -1, Shape.Keep)), neuronCount);
            this.Op = this._conv + this._bias;
        }

        this._initialized = true;
    }
}