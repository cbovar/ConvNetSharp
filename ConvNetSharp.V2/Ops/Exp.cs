using System.Numerics;

namespace ConvNetSharp.V2.Ops;

public class Exp<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public Exp(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
    }

    public Exp(ConvNetSharp<T> graph, Op<T> x) : base(graph)
    {
        this.AddParent(x);
    }

    public override string Representation => "Exp";

    public override void Differentiate()
    {
        this.Parents[0].RegisterDerivative(this.Derivative * this.Graph.Exp(this.Parents[0]));
    }

    public override Volume<T> Evaluate(Session<T> session)
    {
        if (!this.IsDirty)
        {
            return base.Evaluate(session);
        }

        this.IsDirty = false;

        var x = this.Parents[0].Evaluate(session);

        if (this.Result == null || !Equals(this.Result.Shape, x.Shape))
        {
            this.Result?.Dispose();
            this.Result = Volume<T>.CreateEmpty(x.Shape);
        }

        x.Exp(this.Result);
        return base.Evaluate(session);
    }

    public override string ToString()
    {
        return $"Exp({this.Parents[0]})";
    }
}