using System.Numerics;

namespace ConvNetSharp.V2.Ops;

public class DropoutGradient<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    private readonly Dropout<T> _dropout;
    private Shape _lastInputShape;

    public DropoutGradient(ConvNetSharp<T> graph, Dropout<T> dropout, Op<T> derivate) : base(graph)
    {
        this._dropout = dropout;
        this.AddParent(dropout);
        this.AddParent(derivate);
    }

    public override string Representation => "DropoutGradient";

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

        var dropoutOutput = this._dropout.Evaluate(session);
        var dropoutInput = this._dropout.Parents[0].Evaluate(session);
        var dropoutOutputGradient = this._dropout.Derivative.Evaluate(session);
        var dropoutProb = float.CreateChecked(this._dropout.DropoutProbability.Evaluate(session).Get(0)); // TODO: is it correct ?

        if (this.Result == null || !Equals(this._lastInputShape, dropoutInput.Shape))
        {
            this._lastInputShape = new Shape(dropoutInput.Shape);

            this.Result?.Dispose();
            this.Result = Volume<T>.CreateEmpty(dropoutInput.Shape);
        }

        dropoutOutput.DropoutGradient(dropoutInput, dropoutOutputGradient, dropoutProb, this.Result);

        return base.Evaluate(session);
    }
}