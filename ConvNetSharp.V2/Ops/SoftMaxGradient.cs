using System.Numerics;

namespace ConvNetSharp.V2.Ops;

internal class SoftmaxGradient<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    private readonly Softmax<T> _softmax;

    public SoftmaxGradient(ConvNetSharp<T> graph, Softmax<T> softmax) : base(graph)
    {
        this._softmax = softmax;
        this.AddParent(softmax);
    }

    public override string Representation => "SoftmaxGradient";

    public override void Differentiate()
    {
        throw new NotImplementedException();
    }

    public override Volume<T> Evaluate(Session<T> session)
    {
        if (!this.IsDirty)
        {
            return base.Evaluate(session);
        }

        this.IsDirty = false;

        this._softmax.EvaluateGradient(session);

        this.Result = this._softmax.InputGradient;
        return base.Evaluate(session);
    }
}