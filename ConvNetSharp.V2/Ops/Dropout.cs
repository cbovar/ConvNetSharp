using System.Numerics;

namespace ConvNetSharp.V2.Ops;

public class Dropout<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    private Shape _lastInputShape;

    public Dropout(ConvNetSharp<T> graph, Op<T> x, Op<T> dropoutProbability) : base(graph)
    {
        this.AddParent(x);
        this.AddParent(dropoutProbability);
    }

    public Dropout(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
    }

    public override string Representation => $"Dropout({this.DropoutProbability})";

    public Op<T> DropoutProbability => this.Parents[1];

    public override void Differentiate()
    {
        this.Parents[0].RegisterDerivative(new DropoutGradient<T>(this.Graph, this, this.Derivative));
    }

    public override Volume<T> Evaluate(Session<T> session)
    {
        if (!this.IsDirty)
        {
            return base.Evaluate(session);
        }

        this.IsDirty = false;

        var x = this.Parents[0].Evaluate(session);
        var dropoutProb = float.CreateChecked(this.Parents[1].Evaluate(session).Get(0)); // TODO: is it correct ?

        if (this.Result == null || !Equals(this._lastInputShape, x.Shape))
        {
            this._lastInputShape = new Shape(x.Shape);

            this.Result?.Dispose();
            this.Result = Volume<T>.CreateEmpty(x.Shape);
        }

        x.Dropout(dropoutProb, this.Result);

        return base.Evaluate(session);
    }

    public override Dictionary<string, object> GetData()
    {
        var data = base.GetData();
        data["DropoutProbability"] = this.DropoutProbability;
        return data;
    }

    public override string ToString()
    {
        return $"Dropout({this.Parents[0]})";
    }
}