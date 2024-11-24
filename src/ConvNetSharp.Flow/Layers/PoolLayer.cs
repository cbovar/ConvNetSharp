using System.Numerics;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers;

public class PoolLayer<T>(int width, int height) : LayerBase<T>
    where T : INumber<T>, IMinMaxValue<T>
{
    private bool _initialized;
    private int _pad;
    private Pool<T> _pool;
    private int _stride = 1;

    public int Stride
    {
        get => this._stride;
        set
        {
            this._stride = value;
            if (this._initialized)
            {
                this._pool.HorizontalStride = value;
                this._pool.VerticalStride = value;
            }
        }
    }

    public int Pad
    {
        get => this._pad;
        set
        {
            this._pad = value;
            if (this._initialized)
            {
                this._pool.HorizontalPad = value;
                this._pool.VerticalPad = value;
            }
        }
    }

    public override void AcceptParent(LayerBase<T> parent)
    {
        base.AcceptParent(parent);
        this._pool = parent.Op.Graph.Pool(parent.Op, width, height, this.Pad, this.Pad, this.Stride,
            this.Stride);
        this.Op = this._pool;

        this._initialized = true;
    }
}