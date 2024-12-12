using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class MatMultTests
{
    [SetUp]
    public void SetUp()
    {
        this._graph = new ConvNetSharp<float>();
    }

    private ConvNetSharp<float> _graph;

    [Test]
    public void MatMult_ShouldMultiplyTwoScalars()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "scalar_left");
        var right = new Const<float>(this._graph, 2.0f, "scalar_right");
        var matMult = new MatMult<float>(this._graph, left, right);

        // Act
        var result = matMult.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo(new[] { 6.0f }));
    }

    [Test]
    public void MatMult_ShouldHandleMatrixMultiplication()
    {
        // Arrange
        var leftData = new[]
        {
            1.0f, 2.0f,
            3.0f, 4.0f
        }; // 2x2 matrix

        var rightData = new[]
        {
            5.0f, 6.0f,
            7.0f, 8.0f
        }; // 2x2 matrix

        var left = new Const<float>(this._graph, Volume<float>.Create(leftData, new Shape(2, 2)), "left_matrix");
        var right = new Const<float>(this._graph, Volume<float>.Create(rightData, new Shape(2, 2)), "right_matrix");

        var matMult = new MatMult<float>(this._graph, left, right);

        // Act
        var result = matMult.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo(new[]
        {
            19.0f, 22.0f,
            43.0f, 50.0f
        }));
    }

    /// <summary>
    ///     Compute expected 2D matrix multiplication result shape
    ///     [M, K, 1, BatchSize] x [K, N, 1, BatchSize] => [M, N, 1, BatchSize]
    /// </summary>
    /// <param name="leftShape">Left matrix shape</param>
    /// <param name="rightShape">Right matrix shape</param>
    /// <returns>The resulting shape of the matrix multiplication</returns>
    /// <exception cref="ArgumentException">Thrown when shapes are incompatible</exception>
    public static Shape ComputeMatMultiplyShape(Shape leftShape, Shape rightShape)
    {
        // Validate matrix multiplication compatibility
        if (leftShape.Dimensions[1] != rightShape.Dimensions[0])
        {
            throw new ArgumentException($"Matrix dimensions are incompatible for multiplication. " +
                                        $"Left: {leftShape}, Right: {rightShape}. " +
                                        $"Expected Left.Columns == Right.Rows.");
        }

        // Determine batch size using broadcasting rules
        var batchSize = Math.Max(leftShape.Dimensions[3], rightShape.Dimensions[3]);
        if (leftShape.Dimensions[3] != 1 && rightShape.Dimensions[3] != 1 &&
            leftShape.Dimensions[3] != rightShape.Dimensions[3])
        {
            throw new ArgumentException($"Batch sizes are incompatible for broadcasting. " +
                                        $"Left: {leftShape.Dimensions[3]}, Right: {rightShape.Dimensions[3]}.");
        }

        return new Shape(
            leftShape.Dimensions[0], // Rows of the left matrix
            rightShape.Dimensions[1], // Columns of the right matrix
            1, // Depth remains 1
            batchSize // Broadcasted batch size
        );
    }


    [Test]
    public void MatMult_ShouldThrowOnShapeMismatch()
    {
        // Arrange
        var leftData = new[]
        {
            1.0f, 2.0f
        }; // 1x2 matrix

        var rightData = new[]
        {
            3.0f, 4.0f, 5.0f
        }; // 1x3 matrix

        var left = new Const<float>(this._graph, Volume<float>.Create(leftData, new Shape(2, 1)), "invalid_left");
        var right = new Const<float>(this._graph, Volume<float>.Create(rightData, new Shape(3, 1)), "invalid_right");

        var matMult = new MatMult<float>(this._graph, left, right);

        // Act & Assert
        Assert.That(() => matMult.Evaluate(new Session<float>()), Throws.TypeOf<ArgumentException>()
            .With.Message.Contains("Matrix dimensions are incompatible for multiplication"));
    }


    [Test]
    public void MatMult_ShouldDifferentiateCorrectly()
    {
        // Arrange
        var leftData = new[]
        {
            1.0f, 2.0f,
            3.0f, 4.0f
        }; // 2x2 matrix

        var rightData = new[]
        {
            5.0f, 6.0f,
            7.0f, 8.0f
        }; // 2x2 matrix

        var gradientData = new[]
        {
            1.0f, 1.0f,
            1.0f, 1.0f
        }; // Gradient for the result

        var left = new Const<float>(this._graph, Volume<float>.Create(leftData, new Shape(2, 2)), "left");
        var right = new Const<float>(this._graph, Volume<float>.Create(rightData, new Shape(2, 2)), "right");
        var gradient = new Const<float>(this._graph, Volume<float>.Create(gradientData, new Shape(2, 2)), "gradient");

        var matMult = new MatMult<float>(this._graph, left, right);
        matMult.Derivative = gradient;

        // Act
        matMult.Differentiate();

        var leftGradient = left.Derivative.Evaluate(new Session<float>());
        var rightGradient = right.Derivative.Evaluate(new Session<float>());

        // Assert
        Assert.That(leftGradient.ToArray(), Is.EqualTo(new[]
        {
            11.0f, 15.0f,
            11.0f, 15.0f
        }));

        Assert.That(rightGradient.ToArray(), Is.EqualTo(new[]
        {
            4.0f, 4.0f,
            6.0f, 6.0f
        }));
    }





    [Test]
    public void MatMult_ToString_ShouldReturnReadableRepresentation()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "scalar_left");
        var right = new Const<float>(this._graph, 2.0f, "scalar_right");
        var matMult = new MatMult<float>(this._graph, left, right);

        // Act
        var representation = matMult.ToString();

        // Assert
        Assert.That(representation, Is.EqualTo("scalar_left * scalar_right"));
    }

    [Test]
    public void ComputeMatMultiplyShape_ShouldReturnCorrectShape()
    {
        // Arrange
        var leftShape = new Shape(3, 2, 1, 1); // 3 columns, 2 rows
        var rightShape = new Shape(2, 3, 1, 1); // 2 columns, 3 rows

        // Act
        var resultShape = MatMult<float>.ComputeMatMultiplyShape(leftShape, rightShape);

        // Assert
        Assert.That(resultShape, Is.EqualTo(new Shape(2, 2, 1, 1)));
    }
    
    [Test]
    public void ComputeMatMultiplyShape_ShouldThrowOnDimensionMismatch()
    {
        // Arrange
        var leftShape = new Shape(4, 3, 1, 1); // 3 rows, 4 columns
        var rightShape = new Shape(5, 2, 1, 1); // 2 rows, 5 columns

        // Act & Assert
        Assert.Throws<ArgumentException>(() => MatMult<float>.ComputeMatMultiplyShape(leftShape, rightShape));
    }

    [Test]
    public void ComputeMatMultiplyShape_ShouldBroadcastBatchSize()
    {
        // Arrange
        var leftShape = new Shape(4, 3, 1, 1); // 4x3 matrix with batch size 1
        var rightShape = new Shape(2, 4, 1, 5); // 2x4 matrix with batch size 5

        // Act
        var resultShape = MatMult<float>.ComputeMatMultiplyShape(leftShape, rightShape);

        // Assert
        Assert.That(resultShape, Is.EqualTo(new Shape(2, 3, 1, 5))); // Correct result
    }


    [Test]
    public void ComputeMatMultiplyShape_ShouldThrowOnBatchSizeMismatch()
    {
        // Arrange
        var leftShape = new Shape(3, 4, 1, 2); // 4 rows, 3 columns, batch size 2
        var rightShape = new Shape(2, 3, 1, 3); // 3 rows, 2 columns, batch size 3

        // Act & Assert
        Assert.Throws<ArgumentException>(() => MatMult<float>.ComputeMatMultiplyShape(leftShape, rightShape));
    }
}