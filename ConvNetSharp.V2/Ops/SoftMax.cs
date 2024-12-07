using System.Numerics;

namespace ConvNetSharp.V2.Ops;

public class Softmax<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    private long _lastGradientComputeStep = -1;

    public Softmax(ConvNetSharp<T> graph, Op<T> x) : base(graph)
    {
        this.AddParent(x);
    }

    public Softmax(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
    }

    public override string Representation => "Softmax";

    public Volume<T> InputGradient { get; set; }

    public override void Differentiate()
    {
        this.Parents[0].RegisterDerivative(new SoftmaxGradient<T>(this.Graph, this));
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

        x.Softmax(this.Result);

        return base.Evaluate(session);
    }

    public void EvaluateGradient(Session<T> session)
    {
        if (this._lastGradientComputeStep == session.Step)
        {
            return;
        }

        this._lastGradientComputeStep = session.Step;

        var x = this.Parents[0].Evaluate(session);

        if (this.InputGradient == null || !Equals(x.Shape, this.InputGradient.Shape))
        {
            this.InputGradient = Volume<T>.CreateEmpty(x.Shape);
        }

        if (this.Derivative != null)
        {
            x.SoftmaxGradient(this.Derivative.Evaluate(session), this.InputGradient);
        }
    }

    public override string ToString()
    {
        return $"Softmax({this.Parents[0]})";
    }
}