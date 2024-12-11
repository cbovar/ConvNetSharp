using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class MultTests
{
    [SetUp]
    public void SetUp()
    {
        this._graph = new ConvNetSharp<float>();
    }

    private ConvNetSharp<float> _graph;

    [Test]
    public void Mult_ShouldMultiplyTwoScalars()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "left");
        var right = new Const<float>(this._graph, 2.0f, "right");
        var mult = new Mult<float>(this._graph, left, right);

        // Act
        var result = mult.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo(new[] { 6.0f }));
    }

    [Test]
    public void Mult_ShouldBroadcastAndMultiply()
    {
        // Arrange
        var leftVolume = Volume<float>.Create(Enumerable.Repeat(3.0f, 4).ToArray(), new Shape(2, 2));
        var left = new Const<float>(this._graph, leftVolume, "matrix_left");

        var rightVolume = Volume<float>.Create(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new Shape(2, 2));
        var right = new Const<float>(this._graph, rightVolume, "matrix_right");

        var mult = new Mult<float>(this._graph, left, right);

        // Act
        var result = mult.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo(new[] { 3.0f, 6.0f, 9.0f, 12.0f }));
    }

    [Test]
    public void Mult_ShouldHandleDifferentShapesWithBroadcasting()
    {
        // Arrange
        var left = new Const<float>(this._graph, Volume<float>.Create(new[] { 1.0f, 2.0f }, new Shape(2, 1)),
            "vector_left");
        var right = new Const<float>(this._graph,
            Volume<float>.Create(new[] { 3.0f, 4.0f, 5.0f, 6.0f }, new Shape(2, 2)),
            "matrix_right");
        var mult = new Mult<float>(this._graph, left, right);

        // Act
        var result = mult.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo(new[] { 3.0f, 4.0f, 10.0f, 12.0f }));
    }

    [Test]
    public void Mult_ShouldThrowWhenShapesCannotBroadcast()
    {
        // Arrange
        var left = new Const<float>(this._graph, Volume<float>.Create(new[] { 1.0f, 2.0f }, new Shape(2, 1)),
            "invalid_left");
        var right = new Const<float>(this._graph, Volume<float>.Create(new[] { 3.0f, 4.0f, 5.0f }, new Shape(3, 1)),
            "invalid_right");
        var mult = new Mult<float>(this._graph, left, right);

        // Act & Assert
        Assert.That(() => mult.Evaluate(new Session<float>()), Throws.TypeOf<ArgumentException>()
            .With.Message.Contains("Shapes are incompatible for broadcasting."));
    }

    [Test]
    public void Mult_Dispose_ShouldDisposeResult()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "scalar_left");
        var right = new Const<float>(this._graph, 2.0f, "scalar_right");
        var mult = new Mult<float>(this._graph, left, right);

        // Act
        mult.Evaluate(new Session<float>());
        mult.Dispose();

        // Assert
        Assert.That(mult.Result, Is.Null);
    }

    [Test]
    public void Mult_ToString_ShouldReturnReadableRepresentation()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "left");
        var right = new Const<float>(this._graph, 2.0f, "right");
        var mult = new Mult<float>(this._graph, left, right);

        // Act
        var representation = mult.ToString();

        // Assert
        Assert.That(representation, Is.EqualTo("left * right"));
    }

    [Test]
    public void Mult_ShouldDifferentiateCorrectly()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "left");
        var right = new Const<float>(this._graph, 2.0f, "right");
        var mult = new Mult<float>(this._graph, left, right);

        // Act
        mult.Derivative = new Const<float>(this._graph, 1.0f, "dL/dMult");
        mult.Differentiate();

        // Assert
        Assert.That(left.Derivative, Is.Not.Null);
        Assert.That(right.Derivative, Is.Not.Null);

        var leftDerivativeResult = left.Derivative.Evaluate(new Session<float>());
        var rightDerivativeResult = right.Derivative.Evaluate(new Session<float>());

        Assert.That(leftDerivativeResult.ToArray(), Is.EqualTo(new[] { 2.0f })); // Derivative w.r.t left
        Assert.That(rightDerivativeResult.ToArray(), Is.EqualTo(new[] { 3.0f })); // Derivative w.r.t right
    }
}