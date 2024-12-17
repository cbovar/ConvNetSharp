using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class ShapeTests
{
    [Test]
    public void GuessUnknownDimension_ShouldInferMissingDimension()
    {
        // Arrange
        var shape = new Shape(2, -1);

        // Act
        shape.GuessUnknownDimension(10);

        // Assert
        Assert.That(shape.Dimensions[1], Is.EqualTo(5));
    }

    [Test]
    public void GuessUnknownDimension_ShouldThrowForIncompatibleTotalLength()
    {
        // Arrange
        var shape = new Shape(2, -1);

        // Act & Assert
        Assert.That(() => shape.GuessUnknownDimension(9), Throws.ArgumentException);
    }

    [Test]
    public void GuessUnknownDimension_ShouldThrowForNegativeTotalLength()
    {
        // Arrange
        var shape = new Shape(2, -1);

        // Act & Assert
        Assert.That(() => shape.GuessUnknownDimension(-1), Throws.ArgumentException);
    }

    [Test]
    public void SetDimension_ShouldUpdateDimensionAndRecalculateTotalLength()
    {
        // Arrange
        var shape = new Shape(2, 2);

        // Act
        shape.SetDimension(0, 1);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(shape.Dimensions[0], Is.EqualTo(1));
            Assert.That(shape.Dimensions[1], Is.EqualTo(2));
            Assert.That(shape.TotalLength, Is.EqualTo(2));
        });
    }

    [Test]
    public void Constructor_ShouldThrowForZeroDimension()
    {
        // Act & Assert
        Assert.That(() => new Shape(2, 0), Throws.ArgumentException.With.Message.Contains("Dimension cannot be 0"));
    }

    [Test]
    public void Constructor_ShouldThrowForIncorrectDimensionArrayLength()
    {
        // Arrange
        var invalidDimensions = new[] { 1, 2, 3 }; // Only 3 dimensions

        // Act & Assert
        Assert.That(() => new Shape(invalidDimensions),
            Throws.ArgumentException.With.Message.Contains("must contain exactly 4 elements"));
    }

    [Test]
    public void Constructor_ShouldCorrectlyInitializeWithValidDimensions()
    {
        // Arrange
        var validDimensions = new[] { 2, 3, 4, 5 };

        // Act
        var shape = new Shape(validDimensions);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(shape.Dimensions, Is.EqualTo(validDimensions));
            Assert.That(shape.TotalLength, Is.EqualTo(2 * 3 * 4 * 5));
        });
    }

    [Test]
    public void IsVector_ShouldReturnTrueForValidVectorShape()
    {
        // Arrange
        var shape = new Shape(3, 1, 1, 1);

        // Act & Assert
        Assert.That(shape.IsVector, Is.True);
    }

    [Test]
    public void IsMatrix_ShouldReturnTrueForValidMatrixShape()
    {
        // Arrange
        var shape = new Shape(3, 2, 1, 1);

        // Act & Assert
        Assert.That(shape.IsMatrix, Is.True);
    }

    [Test]
    public void IsScalar_ShouldReturnTrueForValidScalarShape()
    {
        // Arrange
        var shape = new Shape(1, 1, 1, 1);

        // Act & Assert
        Assert.That(shape.IsScalar, Is.True);
    }

    [Test]
    public void From_ShouldCreateShapeWithValidDimensions()
    {
        // Act
        var shape = Shape.From(2, 3);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(shape.Dimensions, Is.EqualTo(new[] { 2, 3, 1, 1 }));
            Assert.That(shape.TotalLength, Is.EqualTo(6));
        });
    }

    [Test]
    public void From_ShouldThrowForZeroInDimensions()
    {
        // Act & Assert
        Assert.That(() => Shape.From(0, 3),
            Throws.ArgumentException.With.Message.Contains("Dimensions cannot contain 0"));
    }

    [Test]
    public void From_ShouldThrowForTooManyDimensions()
    {
        // Act & Assert
        Assert.That(() => Shape.From(1, 2, 3, 4, 5),
            Throws.ArgumentException.With.Message.Contains("must have between 1 and 4 elements"));
    }

    [Test]
    public void Equals_ShouldReturnTrueForEqualShapes()
    {
        // Arrange
        var shape1 = new Shape(2, 3, 4, 5);
        var shape2 = new Shape(2, 3, 4, 5);

        // Act & Assert
        Assert.That(shape1.Equals(shape2), Is.True);
    }

    [Test]
    public void Equals_ShouldReturnFalseForUnequalShapes()
    {
        // Arrange
        var shape1 = new Shape(2, 3, 4, 5);
        var shape2 = new Shape(1, 3, 4, 5);

        // Act & Assert
        Assert.That(shape1.Equals(shape2), Is.False);
    }

    [Test]
    public void ToString_ShouldReturnCorrectStringRepresentation()
    {
        // Arrange
        var shape = new Shape(2, 3, 4, 5);

        // Act
        var result = shape.ToString();

        // Assert
        Assert.That(result, Is.EqualTo("2x3x4x5"));
    }

    [Test]
    public void PrettyPrint_ShouldReturnJsonRepresentation()
    {
        // Arrange
        var shape = new Shape(2, 3, 4, 5);

        // Act
        var result = shape.PrettyPrint(asJson: true);

        // Assert
        Assert.That(result, Is.EqualTo("[2, 3, 4, 5]"));
    }

    [Test]
    public void From_ShouldCreateCopyOfExistingShape()
    {
        // Arrange
        var original = new Shape(2, 3, 4, 5);

        // Act
        var copiedShape = Shape.From(original);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(copiedShape, Is.Not.SameAs(original)); // Ensure it's a different instance
            Assert.That(copiedShape.Dimensions, Is.EqualTo(original.Dimensions));
            Assert.That(copiedShape.TotalLength, Is.EqualTo(original.TotalLength));
        });
    }

    [Test]
    public void SetDimension_ShouldUpdateCorrectDimension()
    {
        // Arrange
        var shape = new Shape(2, 3, 4, 5);

        // Act
        shape.SetDimension(2, 6); // Update the third dimension

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(shape.Dimensions[2], Is.EqualTo(6));
            Assert.That(shape.TotalLength, Is.EqualTo(2 * 3 * 6 * 5));
        });
    }

    [Test]
    public void SetDimension_ShouldThrowForInvalidIndex()
    {
        // Arrange
        var shape = new Shape(2, 3, 4, 5);

        // Act & Assert
        Assert.That(() => shape.SetDimension(-1, 6), Throws.TypeOf<ArgumentOutOfRangeException>());
        Assert.That(() => shape.SetDimension(4, 6), Throws.TypeOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public void SetDimension_ShouldHandleEdgeCaseOfChangingAllToOne()
    {
        // Arrange
        var shape = new Shape(2, 2, 2, 2);

        // Act
        for (var i = 0; i < shape.Dimensions.Length; i++)
        {
            shape.SetDimension(i, 1);
        }

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(shape.Dimensions, Is.EqualTo(new[] { 1, 1, 1, 1 }));
            Assert.That(shape.TotalLength, Is.EqualTo(1));
        });
    }

    [Test]
    public void GetStrides_ShouldReturnCorrectStrides()
    {
        // Arrange
        var shape = new Shape(2, 3, 4, 5);

        // Act
        var strides = shape.GetStrides();

        // Assert
        Assert.That(strides, Is.EqualTo(new[] { 60, 20, 5, 1 }));
    }

    [Test]
    public void GetBroadcastedShape_ShouldReturnCorrectShape_WhenBroadcastable()
    {
        // Arrange
        var shape1 = new Shape(2, 3, 4, 5);
        var shape2 = new Shape(1, 3, 1, 1);

        // Act
        var result = Shape.GetBroadcastedShape(shape1, shape2);

        // Assert
        Assert.That(result.Dimensions, Is.EqualTo(new[] { 2, 3, 4, 5 }));
    }

    [Test]
    public void GetBroadcastedShape_ShouldThrowForIncompatibleShapes()
    {
        // Arrange
        var shape1 = new Shape(2, 3, 4, 5);
        var shape2 = new Shape(1, 2, 1, 1);

        // Act & Assert
        Assert.That(() => Shape.GetBroadcastedShape(shape1, shape2),
            Throws.ArgumentException.With.Message.Contains("Shapes are not broadcastable."));
    }

    [Test]
    public void GetCoordinates_ShouldReturnCorrectCoordinates()
    {
        // Arrange
        var shape = new Shape(2, 3, 4, 5);
        var strides = shape.GetStrides();
        var flatIndex = 85; // This corresponds to [1, 1, 1, 0]

        // Act
        var coordinates = shape.GetCoordinates(flatIndex, strides);

        // Assert
        Assert.That(coordinates, Is.EqualTo(new[] { 1, 1, 1, 0 }));
    }

    [Test]
    public void GetBroadcastCoordinates_ShouldReturnCorrectBroadcastedCoordinates()
    {
        // Arrange
        var shape = new Shape(1, 3, 1, 5); // Current shape
        var targetCoordinates = new[] { 0, 2, 0, 4 }; // Coordinates in the broadcasted shape
        var strides = new[] { 15, 5, 5, 1 }; // Strides of the broadcasted shape

        // Act
        var broadcastCoordinates = shape.GetBroadcastCoordinates(targetCoordinates, strides);

        // Assert
        Assert.That(broadcastCoordinates, Is.EqualTo(new[] { 0, 2, 0, 4 })); // Expected broadcasted coordinates
    }

    [Test]
    public void GetFlatIndex_ShouldReturnCorrectFlatIndex()
    {
        // Arrange
        var shape = new Shape(2, 3, 4, 5);
        var coordinates = new[] { 1, 1, 1, 0 };

        // Act
        var flatIndex = shape.GetFlatIndex(coordinates);

        // Assert
        Assert.That(flatIndex, Is.EqualTo(85)); // Expected flat index
    }

    [Test]
    public void GetFlatIndex_ShouldThrowForInvalidCoordinates()
    {
        // Arrange
        var shape = new Shape(2, 3, 4, 5);
        var invalidCoordinates = new[] { 2, 3, 4, 5 }; // Out of bounds

        // Act & Assert
        Assert.That(() => shape.GetFlatIndex(invalidCoordinates), Throws.TypeOf<IndexOutOfRangeException>());
    }

    [Test]
    public void IsBroadcastableTo_ShouldReturnTrueForValidBroadcast()
    {
        // Arrange
        var shape1 = new Shape(1, 3, 1, 1);
        var shape2 = new Shape(2, 3, 4, 5);

        // Act
        var result = shape1.IsBroadcastableTo(shape2);

        // Assert
        Assert.That(result, Is.True);
    }

    [Test]
    public void IsBroadcastableTo_ShouldReturnFalseForInvalidBroadcast()
    {
        // Arrange
        var shape1 = new Shape(1, 3, 2, 1);
        var shape2 = new Shape(2, 3, 4, 5);

        // Act
        var result = shape1.IsBroadcastableTo(shape2);

        // Assert
        Assert.That(result, Is.False);
    }
    
    [Test]
    public void IsBroadcastableTo_ScalarToAnyShape_ShouldReturnTrue()
    {
        // Arrange
        var scalarShape = new Shape(1);
        var targetShape = new Shape(4, 5, 6);

        // Act
        var result = scalarShape.IsBroadcastableTo(targetShape);

        // Assert
        Assert.That(result, Is.True);
    }

    [Test]
    public void IsBroadcastableTo_SameShapes_ShouldReturnTrue()
    {
        // Arrange
        var shape1 = new Shape(3, 4, 5);
        var shape2 = new Shape(3, 4, 5);

        // Act
        var result = shape1.IsBroadcastableTo(shape2);

        // Assert
        Assert.That(result, Is.True);
    }

    [Test]
    public void IsBroadcastableTo_DifferentRankShapes_ShouldReturnFalse()
    {
        // Arrange
        var shape1 = new Shape(3, 1, 1, 1); // Shape [3, 1, 1, 1]
        var shape2 = new Shape(2, 3, 4, 1); // Shape [2, 3, 4, 1]

        Console.WriteLine("Testing broadcastability:");
        Console.WriteLine($"Shape1: {shape1.PrettyPrint(",")}");
        Console.WriteLine($"Shape2: {shape2.PrettyPrint(",")}");

        // Act
        var result = shape1.IsBroadcastableTo(shape2);

        // Assert
        Console.WriteLine($"Broadcastability result: {result}");
        Assert.That(result, Is.False);
    }

    [Test]
    public void IsBroadcastableTo_IncompatibleShapes_ShouldReturnFalse()
    {
        // Arrange
        var shape1 = new Shape(3, 2);
        var shape2 = new Shape(2, 3, 4);

        // Act
        var result = shape1.IsBroadcastableTo(shape2);

        // Assert
        Assert.That(result, Is.False);
    }
}