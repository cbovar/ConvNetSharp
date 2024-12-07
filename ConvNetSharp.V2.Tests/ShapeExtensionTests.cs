using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class ShapeExtensionTests
{
    [Test]
    public void ToVolume_ShouldConvertShapeToVolume_ForIntegerType()
    {
        // Arrange
        var shape = new Shape(3, 5, 7, 9); // Create a shape with dimensions

        // Act
        var volume = shape.ToVolume<int>();

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(volume.Shape.Dimensions[0], Is.EqualTo(4)); // Volume shape
            Assert.That(volume.Get(0), Is.EqualTo(3)); // Width
            Assert.That(volume.Get(1), Is.EqualTo(5)); // Height
            Assert.That(volume.Get(2), Is.EqualTo(7)); // Channels
            Assert.That(volume.Get(3), Is.EqualTo(9)); // Batch size
        });
    }

    [Test]
    public void ToVolume_ShouldConvertShapeToVolume_ForDoubleType()
    {
        // Arrange
        var shape = new Shape(3, 5, 7, 9);

        // Act
        var volume = shape.ToVolume<double>();

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(volume.Shape.Dimensions[0], Is.EqualTo(4)); // Volume stores dimensions as a flat array
            Assert.That(volume.Get(0), Is.EqualTo(3.0)); // Width
            Assert.That(volume.Get(1), Is.EqualTo(5.0)); // Height
            Assert.That(volume.Get(2), Is.EqualTo(7.0)); // Channels
            Assert.That(volume.Get(3), Is.EqualTo(9.0)); // Batch size
        });
    }

    [Test]
    public void ToVolume_ShouldConvertShapeToVolume_ForFloatType()
    {
        // Arrange
        var shape = new Shape(1, 2, 3, 4); // Create a shape with dimensions

        // Act
        var volume = shape.ToVolume<float>();

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(volume.Shape.Dimensions[0], Is.EqualTo(4)); // Volume shape
            Assert.That(volume.Get(0), Is.EqualTo(1.0f)); // Width
            Assert.That(volume.Get(1), Is.EqualTo(2.0f)); // Height
            Assert.That(volume.Get(2), Is.EqualTo(3.0f)); // Channels
            Assert.That(volume.Get(3), Is.EqualTo(4.0f)); // Batch size
        });
    }

    [Test]
    public void ToVolume_ShouldThrowException_WhenShapeHasNegativeDimensions()
    {
        // Arrange
        var shape = new Shape(3, -5, 7, 9); // Invalid shape with a negative dimension

        // Act & Assert
        Assert.Throws<ArgumentException>(() => shape.ToVolume<double>());
    }
}