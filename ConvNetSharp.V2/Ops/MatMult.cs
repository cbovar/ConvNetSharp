using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     Matrix multiplication
///     y = left * right
/// </summary>
/// <typeparam name="T"></typeparam>
public class MatMult<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public MatMult(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
    }

    public MatMult(ConvNetSharp<T> graph, Op<T> left, Op<T> right) : base(graph)
    {
        this.AddParent(left);
        this.AddParent(right);
    }

    public override string Representation => "*";

    public override void Differentiate()
    {
        // dA = GBᵀ, dB = AᵀG
        this.Parents[0].RegisterDerivative(this.Graph.MatMult(this.Derivative, this.Graph.Transpose(this.Parents[1])));
        this.Parents[1].RegisterDerivative(this.Graph.MatMult(this.Graph.Transpose(this.Parents[0]), this.Derivative));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Result?.Dispose();
        }

        base.Dispose(disposing);
    }

    /// <summary>
    ///     Compute expected 2D matrix multiplication result shape
    ///     [K, M, 1, BatchSize] x [N, K, 1, BatchSize] => [N, M, 1, BatchSize]
    /// </summary>
    /// <param name="leftShape">left 2D matrix / volume</param>
    /// <param name="rightShape">right 2D matrix / volume</param>
    /// <returns></returns>
    public static Shape ComputeMatMultiplyShape(Shape leftShape, Shape rightShape)
    {
        // Validate matrix multiplication dimensions
        if (leftShape.Dimensions[0] != rightShape.Dimensions[1])
        {
            throw new ArgumentException($"Matrix dimensions are incompatible for multiplication. " +
                                        $"Left: {leftShape}, Right: {rightShape}. " +
                                        $"Expected Left.Columns == Right.Rows.");
        }

        // Determine batch size using broadcasting rules
        var batchSize = leftShape.Dimensions[3] == 1 || rightShape.Dimensions[3] == 1
            ? Math.Max(leftShape.Dimensions[3], rightShape.Dimensions[3])
            : leftShape.Dimensions[3] == rightShape.Dimensions[3]
                ? leftShape.Dimensions[3]
                : throw new ArgumentException($"Batch sizes are incompatible for broadcasting. " +
                                              $"Left: {leftShape.Dimensions[3]}, Right: {rightShape.Dimensions[3]}.");

        return new Shape(
            rightShape.Dimensions[0], // Columns of the right matrix
            leftShape.Dimensions[1], // Rows of the left matrix
            1, // Depth is assumed to remain 1
            batchSize // Resulting batch size
        );
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

        var shape = ComputeMatMultiplyShape(left.Shape, right.Shape);

        if (this.Result == null || !Equals(this.Result.Shape, shape))
        {
            this.Result?.Dispose();
            this.Result = Volume<T>.CreateEmpty(shape);
        }

        left.MatMultiply(right, this.Result);

        return base.Evaluate(session);
    }

    public override string ToString()
    {
        var addParenthesis = this.Parents[0].Parents.Any();
        var leftStr = addParenthesis ? $"({this.Parents[0]})" : $"{this.Parents[0]}";

        addParenthesis = this.Parents[1].Parents.Any();
        var rightStr = addParenthesis ? $"({this.Parents[1]})" : $"{this.Parents[1]}";

        return $"{leftStr} * {rightStr}";
    }
}