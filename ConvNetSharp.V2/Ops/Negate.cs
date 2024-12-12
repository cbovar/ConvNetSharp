using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     Represents the operation y = -x.
/// </summary>
/// <typeparam name="T">The numeric type of the data (e.g., float, double).</typeparam>
public class Negate<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public Negate(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
    }

    public Negate(ConvNetSharp<T> graph, Op<T> x) : base(graph)
    {
        this.AddParent(x);
    }

    public override string Representation => "Neg";

    /// <summary>
    ///     Computes the derivative of the negate operation.
    /// </summary>
    public override void Differentiate()
    {
        this.Parents[0].RegisterDerivative(-this.Derivative); // Use Negate to compute the negative derivative
    }

    /// <summary>
    ///     Releases resources used by the Negate operation.
    /// </summary>
    /// <param name="disposing">Indicates whether managed resources should also be disposed.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Result?.Dispose();
            this.Result = null; // Ensure Result is set to null
        }

        base.Dispose(disposing);
    }

    /// <summary>
    ///     Evaluates the negate operation.
    /// </summary>
    /// <param name="session">The session to use for evaluation.</param>
    /// <returns>The result of negating the input.</returns>
    public override Volume<T> Evaluate(Session<T> session)
    {
        if (!this.IsDirty)
        {
            return base.Evaluate(session);
        }

        this.IsDirty = false;

        var input = this.Parents[0].Evaluate(session);

        if (this.Result == null || !Equals(this.Result.Shape, input.Shape))
        {
            this.Result?.Dispose();
            this.Result = Volume<T>.CreateEmpty(input.Shape);
        }

        input.Negate(this.Result); // Negate the input and store it in the result volume

        return base.Evaluate(session);
    }

    /// <summary>
    ///     Returns a string representation of the Negate operation.
    /// </summary>
    /// <returns>A string representation of the operation.</returns>
    public override string ToString()
    {
        var addParenthesis = this.Parents[0].Parents.Any();
        return addParenthesis ? $"(-{this.Parents[0]})" : $"-{this.Parents[0]}";
    }
}