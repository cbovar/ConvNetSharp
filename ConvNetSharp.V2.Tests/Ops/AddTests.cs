using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class AddTests
{
    [SetUp]
    public void SetUp()
    {
        this._graph = new ConvNetSharp<float>();
    }

    private ConvNetSharp<float> _graph;

    [Test]
    public void Add_ShouldSumTwoScalars()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "left");
        var right = new Const<float>(this._graph, 2.0f, "right");
        var add = new Add<float>(this._graph, left, right);

        // Act
        var result = add.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo((float[]) [5.0f]));
    }

    [Test]
    public void Add_ShouldBroadcastAndSum()
    {
        // Arrange
        var leftVolume = Volume<float>.Create(Enumerable.Repeat(3.0f, 4).ToArray(), new Shape(2, 2));
        var left = new Const<float>(this._graph, leftVolume, "matrix_left");

        var rightVolume = Volume<float>.Create([1.0f, 2.0f, 3.0f, 4.0f], new Shape(2, 2));
        var right = new Const<float>(this._graph, rightVolume, "matrix_right");

        var add = new Add<float>(this._graph, left, right);

        // Act
        var result = add.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo(new[] { 4.0f, 5.0f, 6.0f, 7.0f }));
    }

    [Test]
    public void Add_ShouldHandleDifferentShapesWithBroadcasting()
    {
        // Arrange
        var left = new Const<float>(this._graph, Volume<float>.Create(new[] { 1.0f, 2.0f }, new Shape(2, 1)), "vector_left");
        var right = new Const<float>(this._graph, Volume<float>.Create(new[] { 3.0f, 4.0f, 5.0f, 6.0f }, new Shape(2, 2)),
            "matrix_right");
        var add = new Add<float>(this._graph, left, right);

        // Act
        var result = add.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo(new float[] { 4.0f, 5.0f, 7.0f, 8.0f }));

    }


    [Test]
    public void Add_ShouldThrowWhenShapesCannotBroadcast()
    {
        // Arrange
        var left = new Const<float>(this._graph, Volume<float>.Create(new[] { 1.0f, 2.0f }, new Shape(2, 1)), "invalid_left");
        var right = new Const<float>(this._graph, Volume<float>.Create(new[] { 3.0f, 4.0f, 5.0f }, new Shape(3, 1)),
            "invalid_right");
        var add = new Add<float>(this._graph, left, right);

        // Act & Assert
        Assert.That(() => add.Evaluate(new Session<float>()), Throws.TypeOf<ArgumentException>()
            .With.Message.Contains("Shapes are incompatible for broadcasting."));
    }


    [Test]
    public void Add_Dispose_ShouldDisposeResult()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "scalar_left");
        var right = new Const<float>(this._graph, 2.0f, "scalar_right");
        var add = new Add<float>(this._graph, left, right);

        // Act
        add.Evaluate(new Session<float>());
        add.Dispose();

        // Assert
        Assert.That(add.Result, Is.Null);
    }

    [Test]
    public void Add_ToString_ShouldReturnReadableRepresentation()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "left");
        var right = new Const<float>(this._graph, 2.0f, "right");
        var add = new Add<float>(this._graph, left, right);

        // Act
        var representation = add.ToString();

        // Assert
        Assert.That(representation, Is.EqualTo("left + right"));
    }

    [Test]
    public void Add_ShouldDifferentiateCorrectly()
    {
        // Arrange
        var left = new Const<float>(this._graph, 3.0f, "left");
        var right = new Const<float>(this._graph, 2.0f, "right");
        var add = new Add<float>(this._graph, left, right);

        // Act
        add.Derivative = new Const<float>(this._graph, 1.0f, "dL/dAdd");
        add.Differentiate();

        // Assert
        Assert.That(left.Derivative, Is.Not.Null);
        Assert.That(right.Derivative, Is.Not.Null);

        var leftDerivativeResult = left.Derivative.Evaluate(new Session<float>());
        var rightDerivativeResult = right.Derivative.Evaluate(new Session<float>());

        Assert.That(leftDerivativeResult.ToArray(), Is.EqualTo((float[]) [1.0f]));
        Assert.That(rightDerivativeResult.ToArray(), Is.EqualTo((float[]) [1.0f]));
    }
}