using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class ConstTests
{
    [Test]
    public void Constructor_WithScalarValue_ShouldInitializeCorrectly()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        const float scalarValue = 5.0f;
        const string name = "ScalarConst";

        // Act
        var constOp = new Const<float>(graph, scalarValue, name);

        // Assert
        Assert.That(constOp.Graph, Is.EqualTo(graph));
        Assert.That(constOp.Value, Is.EqualTo(scalarValue));
        Assert.That(constOp.Name, Is.EqualTo(name));
        Assert.That(constOp.OutputShape, Is.Null);
        Assert.That(constOp.Result, Is.Null); // Should be evaluated lazily
    }

    [Test]
    public void Constructor_WithVolume_ShouldInitializeCorrectly()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volumeShape = new Shape(2, 2);
        var volume = Volume<float>.Create([1f, 2f, 3f, 4f], volumeShape);
        const string name = "VolumeConst";

        // Act
        var constOp = new Const<float>(graph, volume, name);

        // Assert
        Assert.That(constOp.Graph, Is.EqualTo(graph));
        Assert.That(constOp.Result, Is.EqualTo(volume));
        Assert.That(constOp.Name, Is.EqualTo(name));
        Assert.That(constOp.OutputShape, Is.EqualTo(volumeShape));
    }

    [Test]
    public void Constructor_WithDataDictionary_ShouldDeserializeCorrectly()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var data = new Dictionary<string, object>
        {
            { "Name", "DeserializedConst" },
            { "x", 3.14f },
            { "dim0", "1" },
            { "dim1", "2" },
            { "dim2", "1" },
            { "dim3", "1" }
        };

        // Act
        var constOp = new Const<float>(graph, data);

        // Assert
        Assert.That(constOp.Name, Is.EqualTo("DeserializedConst"));
        Assert.That(constOp.Value, Is.EqualTo(3.14f));
        Assert.That(constOp.OutputShape, Is.EqualTo(new Shape(1, 2, 1, 1)));
    }

    [Test]
    public void Evaluate_WithScalarValue_ShouldCreateScalarVolume()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        const float scalarValue = 2.71f;
        var constOp = new Const<float>(graph, scalarValue, "ScalarConst");

        // Act
        using var session = new Session<float>();
        var result = constOp.Evaluate(session);

        // Assert
        Assert.That(result.Shape, Is.EqualTo(new Shape(1, 1, 1, 1)));
        Assert.That(result.Get(0), Is.EqualTo(scalarValue));
    }

    [Test]
    public void Evaluate_WithVolumeShape_ShouldCreateVolume()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        const float value = 1.23f;
        var shape = new Shape(2, 2);
        var constOp = new Const<float>(graph, new Dictionary<string, object>
        {
            { "Name", "VolumeConst" },
            { "x", value },
            { "dim0", "2" },
            { "dim1", "2" },
            { "dim2", "1" },
            { "dim3", "1" }
        });

        // Act
        using var session = new Session<float>();
        var result = constOp.Evaluate(session);

        // Assert
        Assert.That(result.Shape, Is.EqualTo(shape));
        Assert.That(result.ToArray(), Is.All.EqualTo(value));
    }

    [Test]
    public void GetData_ShouldSerializeCorrectly()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        const float scalarValue = 3.14f;
        const string name = "SerializedConst";
        var constOp = new Const<float>(graph, scalarValue, name);

        // Act
        var data = constOp.GetData();

        // Assert
        Assert.That(data["Name"], Is.EqualTo(name));
        Assert.That(data["x"], Is.EqualTo(scalarValue));
    }

    [Test]
    public void Dispose_ShouldCleanUpResult()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var volume = Volume<float>.Create([1f, 2f, 3f, 4f], new Shape(2, 2));
        var constOp = new Const<float>(graph, volume, "VolumeConst");

        // Act
        constOp.Dispose();

        // Assert
        Assert.That(() => volume.ToArray(), Throws.TypeOf<ObjectDisposedException>());
    }

    [Test]
    public void ToString_ShouldReturnName()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        const string name = "ConstName";
        var constOp = new Const<float>(graph, 1f, name);

        // Act
        var result = constOp.ToString();

        // Assert
        Assert.That(result, Is.EqualTo(name));
    }

    [Test]
    public void Differentiate_ShouldNotThrow()
    {
        // Arrange
        var graph = new ConvNetSharp<float>();
        var constOp = new Const<float>(graph, 5f, "Const");

        // Act & Assert
        Assert.That(() => constOp.Differentiate(), Throws.Nothing);
    }
}