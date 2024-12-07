using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     Element wise division
///     y = left / right
/// </summary>
/// <typeparam name="T"></typeparam>
public class Div<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public Div(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
    }

    public Div(ConvNetSharp<T> graph, Op<T> left, Op<T> right) : base(graph)
    {
        this.AddParent(left);
        this.AddParent(right);
    }

    public override string Representation => "/";

    public override void Differentiate()
    {
        this.Parents[0].RegisterDerivative(this.Derivative / this.Parents[1]);
        this.Parents[1].RegisterDerivative(this.Derivative * -this.Parents[0] / (this.Parents[1] * this.Parents[1]));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Result?.Dispose();
        }

        base.Dispose(disposing);
    }

    public override Volume<T> Evaluate(Session<T> session)
    {
        if (!this.IsDirty)
        {
            return base.Evaluate(session);
        }

        this.IsDirty = false;

        var left = this.Parents[0].Evaluate(session);
        var right = this.Parents[1].Evaluate(session);

        if (this.Result == null || !Equals(this.Result.Shape, left.Shape))
        {
            this.Result?.Dispose();
            this.Result = Volume<T>.CreateEmpty(left.Shape);
        }

        left.Divide(right, this.Result);

        return base.Evaluate(session);
    }

    public override string ToString()
    {
        var addParenthesis = this.Parents[0].Parents.Any();
        var leftStr = addParenthesis ? $"({this.Parents[0]})" : $"{this.Parents[0]}";

        addParenthesis = this.Parents[1].Parents.Any();
        var rightStr = addParenthesis ? $"({this.Parents[1]})" : $"{this.Parents[1]}";

        return $"{leftStr} / {rightStr}";
    }
}