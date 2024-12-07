using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class StorageConsistencyTests
{
    [Test]
    public void Add_ShouldProduceSameResultsForCpuAndGpuStorage()
    {
        // Arrange
        var shape = new Shape(1, 3);
        var data1 = new[] { 1.0f, 2.0f, 3.0f };
        var data2 = new[] { 4.0f, 5.0f, 6.0f };

        // CPU Storage
        var cpuStorage1 = new CpuStorage<float>(data1, shape);
        var cpuStorage2 = new CpuStorage<float>(data2, shape);
        var cpuResultData = new float[data1.Length];
        var cpuResultStorage = new CpuStorage<float>(cpuResultData, shape);

        // GPU Storage
        var gpuStorage1 = new GpuStorage<float>(data1, shape);
        var gpuStorage2 = new GpuStorage<float>(data2, shape);
        var gpuResultData = new float[data1.Length];
        var gpuResultStorage = new GpuStorage<float>(gpuResultData, shape);

        // Act
        cpuStorage1.Add(cpuStorage2, cpuResultStorage);
        gpuStorage1.Add(gpuStorage2, gpuResultStorage);

        // Synchronize GPU to ensure operations are complete
        GpuContext.Instance.Synchronize();

        // Assert
        var cpuResult = cpuResultStorage.GetData();
        var gpuResult = gpuResultStorage.GetData();

        Assert.That(cpuResult, Is.EqualTo(gpuResult), "CPU and GPU results should match.");
    }

    [Test]
    public void Add_ShouldHandleInPlaceAddition_Consistently()
    {
        // Arrange
        var shape = new Shape(1, 3);
        var data1 = new[] { 1.0f, 2.0f, 3.0f };
        var data2 = new[] { 4.0f, 5.0f, 6.0f };

        // CPU Storage
        var cpuStorage1 = new CpuStorage<float>(data1, shape);
        var cpuStorage2 = new CpuStorage<float>(data2, shape);

        // GPU Storage
        var gpuStorage1 = new GpuStorage<float>(data1, shape);
        var gpuStorage2 = new GpuStorage<float>(data2, shape);

        // Act
        cpuStorage1.Add(cpuStorage1, cpuStorage1); // In-place addition
        gpuStorage1.Add(gpuStorage1, gpuStorage1);

        // Synchronize GPU to ensure operations are complete
        GpuContext.Instance.Synchronize();

        // Assert
        var cpuResult = cpuStorage1.GetData();
        var gpuResult = gpuStorage1.GetData();

        Assert.That(cpuResult, Is.EqualTo(gpuResult), "CPU and GPU results should match for in-place addition.");
    }
}