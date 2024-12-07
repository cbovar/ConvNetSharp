using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class VariableTests
{
    [Test]
    public void Constructor_ShouldInitializeProperly()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));
        const string name = "Variable1";
        const bool isLearnable = true;

        // Act
        var variable = new Variable<float>(graph, volume, name, isLearnable);

        // Assert
        Assert.That(variable.Graph, Is.EqualTo(graph));
        Assert.That(variable.Result, Is.EqualTo(volume));
        Assert.That(variable.Name, Is.EqualTo(name));
        Assert.That(variable.IsLearnable, Is.EqualTo(isLearnable));
    }

    [Test]
    public void Constructor_ShouldThrowException_WhenNameIsNull()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));

        // Act & Assert
        Assert.That(() => new Variable<float>(graph, volume, null!),
            Throws.ArgumentNullException.With.Message.Contains("name"));
    }

    [Test]
    public void Constructor_WithData_ShouldDeserializeCorrectly()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var data = new Dictionary<string, object>
        {
            { "Name", "Variable1" },
            { "IsLearnable", "True" }
        };

        // Act
        var variable = new Variable<float>(graph, data);

        // Assert
        Assert.That(variable.Name, Is.EqualTo("Variable1"));
        Assert.That(variable.IsLearnable, Is.True);
    }

    [Test]
    public void GetData_ShouldSerializeCorrectly()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));
        var variable = new Variable<float>(graph, volume, "Variable1", true);

        // Act
        var data = variable.GetData();

        // Assert
        Assert.That(data["Name"], Is.EqualTo("Variable1"));
        Assert.That(data["IsLearnable"], Is.EqualTo(true));
    }

    [Test]
    public void SetValue_ShouldUpdateResultAndMarkDirty()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume1 = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));
        var volume2 = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));
        var variable = new Variable<float>(graph, volume1, "Variable1");

        // Act
        variable.SetValue(volume2);

        // Assert
        Assert.That(variable.Result, Is.EqualTo(volume2));
        Assert.That(variable.IsDirty, Is.True);
    }

    [Test]
    public void Differentiate_ShouldCreateConstantDerivative()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));
        var variable = new Variable<float>(graph, volume, "Variable1");

        // Act
        variable.Differentiate();

        // Assert
        Assert.That(variable.Derivative, Is.Not.Null);
        Assert.That(variable.Derivative, Is.InstanceOf<Const<float>>());
        Assert.That(variable.Derivative.Result.Shape, Is.EqualTo(variable.Result.Shape));
    }

    [Test]
    public void Dispose_ShouldCleanUpVolume()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));
        var variable = new Variable<float>(graph, volume, "Variable1");

        // Act
        variable.Dispose();

        // Assert
        Assert.That(() => volume.GetData(), Throws.InstanceOf<ObjectDisposedException>());
    }

    [Test]
    public void Dispose_ShouldBeSafeToCallMultipleTimes()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));
        var variable = new Variable<float>(graph, volume, "Variable1");

        // Act
        variable.Dispose();
        variable.Dispose(); // Call Dispose multiple times

        // Assert
        Assert.Pass("Dispose was safely called multiple times without exception.");
    }

    [Test]
    public void ToString_ShouldReturnName()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume = Volume<float>.CreateEmpty(new Shape(1, 1, 1, 1));
        var variable = new Variable<float>(graph, volume, "Variable1");

        // Act
        var result = variable.ToString();

        // Assert
        Assert.That(result, Is.EqualTo("Variable1"));
    }
}