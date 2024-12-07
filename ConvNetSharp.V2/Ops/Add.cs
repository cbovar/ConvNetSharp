using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     Represents an addition operation in the computation graph.
///     y = left + right
/// </summary>
/// <typeparam name="T">The numeric type (e.g., float, double).</typeparam>
public class Add<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    /// <summary>
    ///     Deserialization constructor.
    /// </summary>
    public Add(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
    }

    /// <summary>
    ///     Initializes a new addition operation with two parent nodes.
    /// </summary>
    /// <param name="graph">The computation graph.</param>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    public Add(ConvNetSharp<T> graph, Op<T> left, Op<T> right) : base(graph)
    {
        if (left == null || right == null)
        {
            throw new ArgumentNullException(nameof(left), "Operands cannot be null.");
        }

        this.AddParent(left);
        this.AddParent(right);
    }

    /// <summary>
    ///     Gets the string representation of the operation.
    /// </summary>
    public override string Representation => "+";

    /// <summary>
    ///     Computes the derivative of the addition operation.
    ///     d(y)/d(left) = 1, d(y)/d(right) = 1
    /// </summary>
    public override void Differentiate()
    {
        var leftDerivative = this.Graph.Sum(this.Derivative, this.Graph.Shape(this.Parents[0]));
        var rightDerivative = this.Graph.Sum(this.Derivative, this.Graph.Shape(this.Parents[1]));

        this.Parents[0].RegisterDerivative(leftDerivative);
        this.Parents[1].RegisterDerivative(rightDerivative);
    }

    /// <summary>
    ///     Disposes of the resources used by this operation.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Result?.Dispose();
            this.Result = null;
        }

        base.Dispose(disposing);
    }

    /// <summary>
    ///     Evaluates the addition operation by adding the results of its parents.
    /// </summary>
    public override Volume<T> Evaluate(Session<T> session)
    {
        if (!this.IsDirty)
        {
            return base.Evaluate(session);
        }

        this.IsDirty = false;

        var left = this.Parents[0].Evaluate(session);
        var right = this.Parents[1].Evaluate(session);

        if (!left.Shape.IsBroadcastableTo(right.Shape))
        {
            throw new ArgumentException("Shapes are incompatible for broadcasting.");
        }

        var resultShape = Shape.GetBroadcastedShape(left.Shape, right.Shape);

        if (this.Result == null || !Equals(this.Result.Shape, resultShape))
        {
            this.Result?.Dispose();
            this.Result = Volume<T>.CreateEmpty(resultShape);
        }

        // Corrected broadcasting logic
        var leftStrides = left.Shape.GetStrides();
        var rightStrides = right.Shape.GetStrides();
        var resultStrides = this.Result.Shape.GetStrides();

        for (var i = 0; i < this.Result.Shape.TotalLength; i++)
        {
            var resultCoordinates = this.Result.Shape.GetCoordinates(i, resultStrides);
            var leftCoordinates = left.Shape.GetBroadcastCoordinates(resultCoordinates, leftStrides);
            var rightCoordinates = right.Shape.GetBroadcastCoordinates(resultCoordinates, rightStrides);

            var leftValue = left.Get(leftCoordinates);
            var rightValue = right.Get(rightCoordinates);

            this.Result.Set(resultCoordinates, leftValue + rightValue);
        }

        return base.Evaluate(session);
    }



    public override string ToString()
    {
        return $"{this.Parents[0]} + {this.Parents[1]}";
    }
}