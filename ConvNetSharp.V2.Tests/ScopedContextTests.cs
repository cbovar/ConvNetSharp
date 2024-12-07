using Moq;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class ScopedContextTests
{
    [SetUp]
    public void SetUp()
    {
        this._mockScopeManager = new Mock<IScopeManager>();
    }

    private Mock<IScopeManager> _mockScopeManager;

    [Test]
    public void Constructor_ShouldThrowArgumentNullException_WhenNameIsNull()
    {
        Assert.That(() => new ScopedContext<float>(null, this._mockScopeManager.Object),
            Throws.ArgumentNullException.With.Message.Contains("Scope name cannot be null."));
    }

    [Test]
    public void Constructor_ShouldThrowArgumentNullException_WhenScopeManagerIsNull()
    {
        Assert.That(() => new ScopedContext<float>("TestScope", null),
            Throws.ArgumentNullException.With.Message.Contains("Scope manager cannot be null."));
    }

    [Test]
    public void Constructor_ShouldCallRegisterScopeOnScopeManager()
    {
        // Act
        using var scopedContext = new ScopedContext<float>("TestScope", this._mockScopeManager.Object);

        // Assert
        this._mockScopeManager.Verify(m => m.RegisterScope("TestScope"), Times.Once);
    }

    [Test]
    public void Dispose_ShouldCallReleaseScopeOnScopeManager()
    {
        // Arrange
        var scopedContext = new ScopedContext<float>("TestScope", this._mockScopeManager.Object);

        // Act
        scopedContext.Dispose();

        // Assert
        this._mockScopeManager.Verify(m => m.ReleaseScope("TestScope"), Times.Once);
    }

    [Test]
    public void Dispose_ShouldBeIdempotent()
    {
        // Arrange
        var scopedContext = new ScopedContext<float>("TestScope", this._mockScopeManager.Object);

        // Act
        scopedContext.Dispose();
        scopedContext.Dispose();

        // Assert
        this._mockScopeManager.Verify(m => m.ReleaseScope("TestScope"), Times.Once);
    }
}