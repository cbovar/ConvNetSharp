using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class MathHelpersTests
{
    [Test]
    public void Log_ShouldReturnCorrectResult_ForFloat()
    {
        // Arrange
        var value = 10.0f;
        var expected = MathF.Log(value);

        // Act
        var result = MathHelpers.Log(value);

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(1e-6f));
    }

    [Test]
    public void Log_ShouldReturnCorrectResult_ForDouble()
    {
        // Arrange
        var value = 10.0;
        var expected = Math.Log(value);

        // Act
        var result = MathHelpers.Log(value);

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(1e-12));
    }

    [Test]
    public void Log_ShouldThrowNotSupportedException_ForUnsupportedType()
    {
        // Arrange
        var value = 10;

        // Act & Assert
        Assert.Throws<NotSupportedException>(() => MathHelpers.Log(value));
    }

    [Test]
    public void Log_ShouldThrowForNegativeInput_ForFloat()
    {
        // Arrange
        var value = -1.0f;

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => MathHelpers.Log(value));
    }

    [Test]
    public void Log_ShouldThrowForNegativeInput_ForDouble()
    {
        // Arrange
        var value = -1.0;

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => MathHelpers.Log(value));
    }
}