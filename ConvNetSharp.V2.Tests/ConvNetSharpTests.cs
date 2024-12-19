using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class ConvNetSharpTests
{
    [SetUp]
    public void SetUp()
    {
        this._graph = new ConvNetSharp<float>();
    }

    private ConvNetSharp<float> _graph;

    [Test]
    public void Variable_ShouldCreateScopedVariable()
    {
        // Arrange
        var shape = new Shape(1, 1, 10, 10);

        using (this._graph.Scope("layer1"))
        {
            // Act
            var variable = this._graph.Variable(shape, "weights");

            // Assert
            Assert.That(variable.Name, Is.EqualTo("layer1/weights"));
            Assert.That(variable.IsLearnable, Is.False);
        }
    }

    [Test]
    public void Variable_WithLearnable_ShouldSetIsLearnableFlag()
    {
        // Arrange
        var shape = new Shape(1, 1, 10, 10);

        // Act
        var variable = this._graph.Variable(shape, "bias", true);

        // Assert
        Assert.That(variable.IsLearnable, Is.True);
    }

    [Test]
    public void Const_ShouldCreateConstantFromValue()
    {
        // Act
        var constant = this._graph.Const(1.0f, "scalar_const");

        // Assert
        Assert.That(constant.Value, Is.EqualTo(1.0f));
        Assert.That(constant.Name, Is.EqualTo("scalar_const"));
    }

    [Test]
    public void Const_ShouldCreateConstantFromVolume()
    {
        // Arrange
        var volume = Volume<float>.Create([1.0f, 2.0f, 3.0f], new Shape(1, 1, 3, 1));

        // Act
        var constant = this._graph.Const(volume, "volume_const");

        // Assert
        Assert.That(constant.Result, Is.EqualTo(volume));
        Assert.That(constant.Name, Is.EqualTo("volume_const"));
    }

    [Test]
    public void Conv_ShouldCreateConvolutionOperation_WithExplicitFilter()
    {
        // Arrange
        var inputShape = new Shape(28, 28, 3, 1);
        var filterShape = new Shape(3, 3, 3, 16);
        var input = this._graph.PlaceHolder("input");
        var filter = Volume<float>.CreateRandom(filterShape, 0.0, 1.0);

        // Simulate input volume evaluation for the test
        input.Result = Volume<float>.CreateEmpty(inputShape);

        // Act
        var convOp = this._graph.Conv(input, filter, 3, 3, 16);

        // Assert
        Assert.That(convOp, Is.TypeOf<Convolution<float>>());
        Assert.That(convOp.Width, Is.EqualTo(3));
        Assert.That(convOp.Height, Is.EqualTo(3));
        Assert.That(convOp.FilterCount, Is.EqualTo(16));
        Assert.That(convOp.Filter.Result, Is.EqualTo(filter));

        // Simulate evaluation
        using var session = new Session<float>();
        var result = convOp.Evaluate(session);
        Assert.That(result.Shape.Dimensions[0], Is.EqualTo(26));
        Assert.That(result.Shape.Dimensions[1], Is.EqualTo(26));
    }

    [Test]
    public void Conv_ShouldThrowException_WhenFilterIsNull()
    {
        // Arrange
        var inputOp = this._graph.PlaceHolder("input");

        // Act & Assert
        Assert.That(() => this._graph.Conv(inputOp, null!, 3, 3, 16),
            Throws.ArgumentNullException.With.Message.Contains("Filter must be explicitly provided."));
    }

    [Test]
    public void Dense_ShouldCreateFullyConnectedLayer_WithExplicitFilter()
    {
        // Arrange
        var inputShape = new Shape(28, 28, 3, 1);
        var filterShape = new Shape(1, 1, 3, 128);
        var input = Volume<float>.CreateEmpty(inputShape);
        var filter = Volume<float>.CreateRandom(filterShape, 0.0, 1.0);

        // Act
        var inputOp = this._graph.PlaceHolder("input");
        var denseOp = new Dense<float>(this._graph, inputOp, filter);

        // Assert
        Assert.That(denseOp.Width, Is.EqualTo(1));
        Assert.That(denseOp.Height, Is.EqualTo(1));
        Assert.That(denseOp.FilterCount, Is.EqualTo(128));
        Assert.That(denseOp.Filter.Result, Is.EqualTo(filter));
    }

    [Test]
    public void Dense_ShouldThrowException_WhenFilterShapeIsInvalid()
    {
        // Arrange
        var inputOp = this._graph.PlaceHolder("input");
        var invalidFilter = Volume<float>.CreateEmpty(new Shape(3, 3, 3, 128));

        // Act & Assert
        Assert.That(() => new Dense<float>(this._graph, inputOp, invalidFilter),
            Throws.ArgumentException.With.Message.Contains("Dense layer filters must have a shape of 1x1."));
    }
}