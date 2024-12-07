using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     Computes u^v
/// </summary>
/// <typeparam name="T"></typeparam>
public class Power<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public Power(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
    }

    public Power(ConvNetSharp<T> graph, Op<T> u, Op<T> v) : base(graph)
    {
        this.AddParent(u);
        this.AddParent(v);
    }

    public override string Representation => "Pow";

    public override void Differentiate()
    {
        var u = this.Parents[0];
        var v = this.Parents[1];

        // d(u^v)/d(u) = v.u^(v-1)
        u.RegisterDerivative(this.Derivative * v * (u ^ (v - this.Graph.Const(ConvNetSharp<T>.One, "one"))));

        // d(u^v)/d(v) = u^v.log(u)
        v.RegisterDerivative(this.Derivative * (u ^ v) * this.Graph.Log(u));
    }

    public override Volume<T> Evaluate(Session<T> session)
    {
        if (!this.IsDirty)
        {
            return base.Evaluate(session);
        }

        this.IsDirty = false;

        var u = this.Parents[0].Evaluate(session);
        var v = this.Parents[1].Evaluate(session);

        if (this.Result == null || !Equals(this.Result.Shape, u.Shape))
        {
            this.Result?.Dispose();
            this.Result = Volume<T>.CreateEmpty(u.Shape);
        }

        u.Power(v, this.Result);
        return base.Evaluate(session);
    }

    public override string ToString()
    {
        return $"Pow({this.Parents[0]}, {this.Parents[1]})";
    }
}