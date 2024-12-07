using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class PlaceHolderTests
{
    [SetUp]
    public void SetUp()
    {
        this._graph = new ConvNetSharp<float>();
    }

    private ConvNetSharp<float> _graph;

    [Test]
    public void PlaceHolder_ShouldInitializeWithName_FromConstructor()
    {
        // Arrange
        var name = "input";

        // Act
        var placeHolder = new PlaceHolder<float>(this._graph, name);

        // Assert
        Assert.That(placeHolder.Name, Is.EqualTo(name));
        Assert.That(placeHolder.Representation, Is.EqualTo(name));
    }

    [Test]
    public void PlaceHolder_ShouldInitializeWithDataDictionary()
    {
        // Arrange
        var data = new Dictionary<string, object> { { "Name", "input" } };

        // Act
        var placeHolder = new PlaceHolder<float>(this._graph, data);

        // Assert
        Assert.That(placeHolder.Name, Is.EqualTo("input"));
    }

    [Test]
    public void PlaceHolder_ShouldReturnCorrectRepresentation()
    {
        // Arrange
        var name = "test_placeholder";
        var placeHolder = new PlaceHolder<float>(this._graph, name);

        // Act
        var representation = placeHolder.Representation;

        // Assert
        Assert.That(representation, Is.EqualTo(name));
    }

    [Test]
    public void PlaceHolder_SetValue_ShouldUpdateResult()
    {
        // Arrange
        var placeHolder = new PlaceHolder<float>(this._graph, "input");
        var shape = new Shape(3, 3);
        var volume = Volume<float>.Create(new float[9], shape);

        // Act
        placeHolder.SetValue(volume);

        // Assert
        Assert.That(placeHolder.Result, Is.EqualTo(volume));
    }

    [Test]
    public void PlaceHolder_SetValue_ShouldSetDirtyFlag()
    {
        // Arrange
        var placeHolder = new PlaceHolder<float>(this._graph, "input");
        var shape = new Shape(3, 3);
        var volume = Volume<float>.Create(new float[9], shape);

        // Act
        placeHolder.SetValue(volume);

        // Assert
        Assert.That(placeHolder.IsDirty, Is.True);
    }

    [Test]
    public void PlaceHolder_GetData_ShouldIncludeName()
    {
        // Arrange
        var name = "input";
        var placeHolder = new PlaceHolder<float>(this._graph, name);

        // Act
        var data = placeHolder.GetData();

        // Assert
        Assert.That(data.ContainsKey("Name"));
        Assert.That(data["Name"], Is.EqualTo(name));
    }

    [Test]
    public void PlaceHolder_Dispose_ShouldDisposeResult()
    {
        // Arrange
        var placeHolder = new PlaceHolder<float>(this._graph, "input");
        var shape = new Shape(3, 3);
        var volume = Volume<float>.Create(new float[9], shape);
        placeHolder.SetValue(volume);

        // Act
        placeHolder.Dispose();

        // Assert
        Assert.That(placeHolder.Result, Is.Null);
    }

    [Test]
    public void PlaceHolder_ToString_ShouldReturnName()
    {
        // Arrange
        var name = "test_placeholder";
        var placeHolder = new PlaceHolder<float>(this._graph, name);

        // Act
        var result = placeHolder.ToString();

        // Assert
        Assert.That(result, Is.EqualTo(name));
    }
}