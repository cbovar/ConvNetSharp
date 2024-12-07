using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class GpuContextTests
{
    [SetUp]
    public void SetUp()
    {
        // Ensure GpuContext is reset before each test
        GpuContext.Reset();
    }

    [TearDown]
    public void TearDown()
    {
        // Reset the GpuContext after each test
        GpuContext.Reset();
    }

    [Test]
    public void Initialize_ShouldSetIsInitializedToTrue()
    {
        // Act
        GpuContext.Instance.Initialize();

        // Assert
        Assert.That(GpuContext.Instance.IsInitialized, Is.True);
    }

    [Test]
    public void Initialize_ShouldCreateCudaContextAndDefaultStream()
    {
        // Act
        GpuContext.Instance.Initialize();

        // Assert
        Assert.That(GpuContext.Instance.CudaContext, Is.Not.Null);
        Assert.That(GpuContext.Instance.DefaultStream, Is.Not.Null);
    }

    [Test]
    public void Initialize_ShouldLogGpuDetails()
    {
        // Act
        GpuContext.Instance.Initialize();

        // Assert
        var cudaContext = GpuContext.Instance.CudaContext;
        var deviceProperties = cudaContext.GetDeviceInfo();

        Console.WriteLine($"GPU Name: {deviceProperties.DeviceName}");
        Console.WriteLine(
            $"Compute Capability: {deviceProperties.ComputeCapability.Major}.{deviceProperties.ComputeCapability.Minor}");
        Console.WriteLine($"Total Global Memory: {deviceProperties.TotalGlobalMemory / (1024 * 1024)} MB");
        Console.WriteLine($"Multiprocessors: {deviceProperties.MultiProcessorCount}");
    }

    [Test]
    public void Dispose_ShouldCleanupResources()
    {
        // Arrange
        GpuContext.Instance.Initialize();

        // Act
        GpuContext.Instance.Dispose();

        // Assert
        Assert.That(GpuContext.Instance.IsInitialized, Is.False);
        Assert.That(GpuContext.Instance.CudaContext, Is.Null);
        Assert.That(GpuContext.Instance.CublasContext, Is.Null);
        Assert.That(GpuContext.Instance.DefaultStream, Is.Null);
    }

    [Test]
    public void Synchronize_ShouldThrowIfNotInitialized()
    {
        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => GpuContext.Instance.Synchronize());
        Assert.That(ex.Message, Is.EqualTo("GPU context is not initialized."));
    }

    [Test]
    public void Synchronize_ShouldNotThrowIfInitialized()
    {
        // Arrange
        GpuContext.Instance.Initialize();

        // Act & Assert
        Assert.DoesNotThrow(() => GpuContext.Instance.Synchronize());
    }

    [Test]
    public void Reset_ShouldDisposeExistingInstance()
    {
        // Arrange
        GpuContext.Instance.Initialize();

        // Act
        GpuContext.Reset();

        // Assert
        Assert.That(GpuContext.Instance.IsInitialized, Is.False);
    }

    [Test]
    public void ReinitializeAfterReset_ShouldWork()
    {
        // Arrange
        GpuContext.Instance.Initialize();
        GpuContext.Reset();

        // Act
        GpuContext.Instance.Initialize();

        // Assert
        Assert.That(GpuContext.Instance.IsInitialized, Is.True);
        Assert.That(GpuContext.Instance.CudaContext, Is.Not.Null);
        Assert.That(GpuContext.Instance.DefaultStream, Is.Not.Null);
    }

    [Test]
    public void CublasContext_ShouldInitializeProperly()
    {
        // Act
        GpuContext.Instance.Initialize();

        // Assert
        Assert.That(GpuContext.Instance.CublasContext, Is.Not.Null, "CublasContext should be initialized.");
    }
}