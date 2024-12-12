using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class NegateTests
{
    [SetUp]
    public void SetUp()
    {
        this._graph = new ConvNetSharp<float>();
    }

    private ConvNetSharp<float> _graph;

    [Test]
    public void Negate_ShouldNegateScalar()
    {
        // Arrange
        var input = new Const<float>(this._graph, 3.0f, "input");
        var negate = new Negate<float>(this._graph, input);

        // Act
        var result = negate.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo((float[]) [-3.0f]));
    }

    [Test]
    public void Negate_ShouldNegateMatrix()
    {
        // Arrange
        var inputVolume = Volume<float>.Create(new[] { 1.0f, -2.0f, 3.0f, -4.0f }, new Shape(2, 2));
        var input = new Const<float>(this._graph, inputVolume, "matrix_input");
        var negate = new Negate<float>(this._graph, input);

        // Act
        var result = negate.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo(new[] { -1.0f, 2.0f, -3.0f, 4.0f }));
    }

    [Test]
    public void Negate_Dispose_ShouldDisposeResult()
    {
        // Arrange
        var input = new Const<float>(this._graph, 3.0f, "scalar_input");
        var negate = new Negate<float>(this._graph, input);

        // Act
        negate.Evaluate(new Session<float>());
        negate.Dispose();

        // Assert
        Assert.That(negate.Result, Is.Null);
    }

    [Test]
    public void Negate_ToString_ShouldReturnReadableRepresentation()
    {
        // Arrange
        var input = new Const<float>(this._graph, 3.0f, "input");
        var negate = new Negate<float>(this._graph, input);

        // Act
        var representation = negate.ToString();

        // Assert
        Assert.That(representation, Is.EqualTo("-input"));
    }

    [Test]
    public void Negate_ShouldDifferentiateCorrectly()
    {
        // Arrange
        var input = new Const<float>(this._graph, 3.0f, "input");
        var negate = new Negate<float>(this._graph, input);

        // Act
        negate.Derivative = new Const<float>(this._graph, 1.0f, "dL/dNegate");
        negate.Differentiate();

        // Assert
        Assert.That(input.Derivative, Is.Not.Null);

        var inputDerivativeResult = input.Derivative.Evaluate(new Session<float>());

        Assert.That(inputDerivativeResult.ToArray(), Is.EqualTo((float[]) [-1.0f]));
    }

    [Test]
    public void Negate_ShouldHandleNestedNegations()
    {
        // Arrange
        var input = new Const<float>(this._graph, 3.0f, "input");
        var negate1 = new Negate<float>(this._graph, input);
        var negate2 = new Negate<float>(this._graph, negate1);

        // Act
        var result = negate2.Evaluate(new Session<float>());

        // Assert
        Assert.That(result.ToArray(), Is.EqualTo((float[]) [3.0f]));
    }
}