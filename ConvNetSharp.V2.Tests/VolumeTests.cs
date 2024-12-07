using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class VolumeTests
{
    [Test]
    public void CreateCpuVolume_ShouldInitializeCorrectly()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);

        // Act
        var cpuVolume = Volume<double>.Create(data, shape);

        // Assert
        Assert.That(cpuVolume.Shape.TotalLength, Is.EqualTo(shape.TotalLength));
        Assert.That(cpuVolume.GetData(), Is.EqualTo(data));
    }

    [Test]
    public void CreateGpuVolume_ShouldInitializeCorrectly()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);

        // Act
        var gpuVolume = Volume<double>.Create(data, shape, Device.GPU);

        // Assert
        Assert.That(gpuVolume.Shape.TotalLength, Is.EqualTo(shape.TotalLength));
        Assert.That(gpuVolume.GetData(), Is.EqualTo(data));
    }

    [Test]
    public void ApplyOperation_ShouldModifyCpuVolumeData()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var cpuVolume = Volume<double>.Create(data, shape);

        // Act
        cpuVolume.ApplyOperation(x => x * 2);

        // Assert
        Assert.That(cpuVolume.GetData(), Is.EqualTo((double[]) [2.0, 4.0, 6.0]));
    }

    [Test]
    public void ApplyOperation_ShouldModifyGpuVolumeData()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var gpuVolume = Volume<double>.Create(data, shape, Device.GPU);

        // Act
        gpuVolume.ApplyOperation(x => x * 2);

        // Act & Assert
        Assert.That(() => gpuVolume.GetData(), Is.EqualTo((double[]) [2.0, 4.0, 6.0]));
    }

    [Test]
    public void GetData_ShouldReturnCorrectDataForCpuVolume()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var cpuVolume = Volume<double>.Create(data, shape);

        // Act
        var result = cpuVolume.GetData();

        // Assert
        Assert.That(result, Is.EqualTo(data));
    }

    [Test]
    public void GetData_ShouldReturnCorrectDataForGpuVolume()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var gpuVolume = Volume<double>.Create(data, shape, Device.GPU);

        // Act
        var result = gpuVolume.GetData();

        // Assert
        Assert.That(result, Is.EqualTo(data));
    }

    [Test]
    public void CreateVolume_WithInvalidDevice_ShouldThrowException()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);

        // Act & Assert
        Assert.That(() => Volume<double>.Create(data, shape, (Device)999), Throws.TypeOf<ArgumentException>());
    }


    [Test]
    public void Clear_ShouldResetCpuVolumeDataToZero()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var cpuVolume = Volume<double>.Create(data, shape);

        // Act
        cpuVolume.Clear();

        // Assert
        Assert.That(cpuVolume.GetData(), Is.EqualTo(new[] { 0.0, 0.0, 0.0 }));
    }

    [Test]
    public void Clear_ShouldResetGpuVolumeDataToZero()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var gpuVolume = Volume<double>.Create(data, shape, Device.GPU);

        // Act
        gpuVolume.Clear();

        // Assert
        Assert.That(gpuVolume.GetData(), Is.EqualTo(new[] { 0.0, 0.0, 0.0 }));
    }

    [Test]
    public void Add_ShouldAddTwoCpuVolumes()
    {
        // Arrange
        var data1 = new[] { 1.0, 2.0, 3.0 };
        var data2 = new[] { 4.0, 5.0, 6.0 };
        var shape = new Shape(1, 3);
        var cpuVolume1 = Volume<double>.Create(data1, shape);
        var cpuVolume2 = Volume<double>.Create(data2, shape);
        var result = Volume<double>.CreateEmpty(shape);

        // Act
        cpuVolume1.Add(cpuVolume2, result);

        // Assert
        Assert.That(result.GetData(), Is.EqualTo(new[] { 5.0, 7.0, 9.0 }));
    }

    [Test]
    public void Add_ShouldThrowForMismatchedShapes()
    {
        // Arrange
        var data1 = new[] { 1.0, 2.0, 3.0 };
        var data2 = new[] { 4.0, 5.0 };
        var shape1 = new Shape(1, 3);
        var shape2 = new Shape(1, 2);
        var cpuVolume1 = Volume<double>.Create(data1, shape1);
        var cpuVolume2 = Volume<double>.Create(data2, shape2);
        var result = Volume<double>.CreateEmpty(shape1);

        // Act & Assert
        Assert.That(() => cpuVolume1.Add(cpuVolume2, result), Throws.TypeOf<ArgumentException>());
    }

    [Test]
    public void SetAndGet_ShouldWorkCorrectlyForCpuVolume()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var cpuVolume = Volume<double>.Create(data, shape);

        // Act
        cpuVolume.Set(0, 10.0);
        cpuVolume.Set(1, 20.0);

        // Assert
        Assert.That(cpuVolume.Get(0), Is.EqualTo(10.0));
        Assert.That(cpuVolume.Get(1), Is.EqualTo(20.0));
    }

    [Test]
    public void SetAndGet_ShouldWorkCorrectlyForGpuVolume()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var gpuVolume = Volume<double>.Create(data, shape, Device.GPU);

        // Act
        gpuVolume.Set(0, 10.0);
        gpuVolume.Set(1, 20.0);

        // Assert
        Assert.That(gpuVolume.Get(0), Is.EqualTo(10.0));
        Assert.That(gpuVolume.Get(1), Is.EqualTo(20.0));
    }

    [Test]
    public void ReShape_ShouldChangeVolumeShape()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var initialShape = new Shape(1, 6);
        var newShape = new Shape(2, 3);
        var cpuVolume = Volume<double>.Create(data, initialShape);

        // Act
        var reshapedVolume = cpuVolume.ReShape(newShape);

        // Assert
        Assert.That(reshapedVolume.Shape.Dimensions, Is.EqualTo(newShape.Dimensions));
        Assert.That(reshapedVolume.GetData(), Is.EqualTo(data));
    }

    [Test]
    public void ReShape_ShouldThrowForMismatchedTotalLength()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var initialShape = new Shape(1, 3);
        var newShape = new Shape(2, 2);
        var cpuVolume = Volume<double>.Create(data, initialShape);

        // Act & Assert
        Assert.That(() => cpuVolume.ReShape(newShape), Throws.TypeOf<ArgumentException>());
    }

    [Test]
    public void Dispose_ShouldReleaseResources()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var cpuVolume = Volume<double>.Create(data, shape);

        // Act
        cpuVolume.Dispose();

        // Assert
        Assert.That(() => cpuVolume.GetData(), Throws.TypeOf<ObjectDisposedException>());
    }

    [Test]
    public void ReShape_ShouldPreserveKeepDimensions()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var initialShape = new Shape(2, 3); // [Width=2, Height=3]
        var reshapedWithKeep = new Shape(Shape.Keep, 1, 3, 1); // Keep Width, set Height to 1, Channels=3, BatchSize=1
        var cpuVolume = Volume<double>.Create(data, initialShape);

        // Act
        var reshapedVolume = cpuVolume.ReShape(reshapedWithKeep);

        // Assert
        Assert.That(reshapedVolume.Shape.Dimensions[0],
            Is.EqualTo(initialShape.Dimensions[0])); // Width remains unchanged
        Assert.That(reshapedVolume.Shape.Dimensions[1], Is.EqualTo(1)); // Height explicitly set to 1
        Assert.That(reshapedVolume.Shape.Dimensions[2], Is.EqualTo(3)); // Channels explicitly set to 3
        Assert.That(reshapedVolume.Shape.Dimensions[3], Is.EqualTo(1)); // BatchSize explicitly set to 1
        Assert.That(reshapedVolume.GetData(), Is.EqualTo(data)); // Data should remain unchanged
    }

    [Test]
    public void SetAndGet_1DCoordinates_ShouldStoreAndRetrieveValue()
    {
        // Arrange
        var data = new double[6];
        var shape = new Shape(6); // 1D volume
        var volume = Volume<double>.Create(data, shape);

        // Act
        volume.Set(2, 42.0);
        var value = volume.Get(2);

        // Assert
        Assert.That(value, Is.EqualTo(42.0));
    }

    [Test]
    public void SetAndGet_2DCoordinates_ShouldStoreAndRetrieveValue()
    {
        // Arrange
        var data = new double[6];
        var shape = new Shape(2, 3); // 2D volume
        var volume = Volume<double>.Create(data, shape);

        // Act
        volume.Set(1, 2, 42.0);
        var value = volume.Get(1, 2);

        // Assert
        Assert.That(value, Is.EqualTo(42.0));
    }

    [Test]
    public void SetAndGet_3DCoordinates_ShouldStoreAndRetrieveValue()
    {
        // Arrange
        var data = new double[24];
        var shape = new Shape(2, 3, 4); // 3D volume
        var volume = Volume<double>.Create(data, shape);

        // Act
        volume.Set(1, 2, 3, 42.0);
        var value = volume.Get(1, 2, 3);

        // Assert
        Assert.That(value, Is.EqualTo(42.0));
    }

    [Test]
    public void SetAndGet_4DCoordinates_ShouldStoreAndRetrieveValue()
    {
        // Arrange
        var data = new double[48];
        var shape = new Shape(2, 3, 4, 2); // 4D volume
        var volume = Volume<double>.Create(data, shape);

        // Act
        volume.Set(1, 2, 3, 1, 42.0);
        var value = volume.Get(1, 2, 3, 1);

        // Assert
        Assert.That(value, Is.EqualTo(42.0));
    }

    [Test]
    public void SetAndGet_NDCoordinates_ShouldStoreAndRetrieveValue()
    {
        // Arrange
        var data = new double[48];
        var shape = new Shape(2, 3, 4, 2); // 4D volume
        var volume = Volume<double>.Create(data, shape);
        var coordinates = new[] { 1, 2, 3, 1 };

        // Act
        volume.Set(coordinates, 42.0);
        var value = volume.Get(coordinates);

        // Assert
        Assert.That(value, Is.EqualTo(42.0));
    }

    [Test]
    public void ReShape_ParamsArray_ShouldUpdateShapeCorrectly()
    {
        // Arrange
        var data = new double[6]; // Total length = 6
        var originalShape = new Shape(6, 1); // Original shape: [6, 1, 1, 1]
        var volume = Volume<double>.Create(data, originalShape);

        // Act
        var reshapedVolume = volume.ReShape(2, 3); // Target shape: [2, 3, 1, 1]

        // Assert
        Assert.That(reshapedVolume.Shape.Dimensions, Is.EqualTo(new[] { 2, 3, 1, 1 }));
        Assert.That(reshapedVolume.Shape.TotalLength, Is.EqualTo(6)); // Total length should remain the same
    }

    [Test]
    public void CreateRandom_ShouldInitializeWithCorrectShapeAndRandomValues()
    {
        // Arrange
        var shape = new Shape(3, 3); // Shape [3, 3, 1, 1]
        var mean = 0.0;
        var scale = 1.0;

        // Act
        var randomVolume = Volume<double>.CreateRandom(shape, mean, scale);

        // Assert
        Assert.That(randomVolume.Shape.Dimensions, Is.EqualTo(new[] { 3, 3, 1, 1 }));
        Assert.That(randomVolume.Shape.TotalLength, Is.EqualTo(9));

        var data = randomVolume.GetData();

        // Check that data is not uniform (likely randomized) and values are within a plausible range
        Assert.That(data.Distinct().Count(), Is.GreaterThan(1)); // Ensure random values
        Assert.That(data.All(x => x >= mean - 4 * scale && x <= mean + 4 * scale),
            "Values should fall within ~4 standard deviations of the mean");
    }

    [Test]
    public void Ones_ShouldCreateVolumeFilledWithOnes()
    {
        // Arrange
        var shape = new Shape(2, 3); // Shape [2, 3, 1, 1]

        // Act
        var onesVolume = Volume<double>.Ones(shape);

        // Assert
        Assert.That(onesVolume.Shape.Dimensions, Is.EqualTo(new[] { 2, 3, 1, 1 }));
        Assert.That(onesVolume.Shape.TotalLength, Is.EqualTo(6)); // Total length = 2 * 3 * 1 * 1

        var data = onesVolume.GetData();

        // Ensure all values are 1.0
        Assert.That(data, Is.All.EqualTo(1.0));
    }

    [Test]
    public void Zeros_ShouldCreateVolumeFilledWithZeros()
    {
        // Arrange
        var shape = new Shape(3, 3); // Shape [3, 3, 1, 1]

        // Act
        var zerosVolume = Volume<double>.Zeros(shape);

        // Assert
        Assert.That(zerosVolume.Shape.Dimensions, Is.EqualTo(new[] { 3, 3, 1, 1 }));
        Assert.That(zerosVolume.Shape.TotalLength, Is.EqualTo(9)); // Total length = 3 * 3 * 1 * 1

        var data = zerosVolume.GetData();

        // Ensure all values are 0.0
        Assert.That(data, Is.All.EqualTo(0.0));
    }

    [Test]
    public void Ones_ShouldCreateGpuVolumeFilledWithOnes()
    {
        // Arrange
        var shape = new Shape(4, 4); // Shape [4, 4, 1, 1]

        // Act
        var onesGpuVolume = Volume<double>.Ones(shape);

        // Assert
        Assert.That(onesGpuVolume.Shape.Dimensions, Is.EqualTo(new[] { 4, 4, 1, 1 }));
        Assert.That(onesGpuVolume.Shape.TotalLength, Is.EqualTo(16)); // Total length = 4 * 4 * 1 * 1

        var data = onesGpuVolume.GetData();

        // Ensure all values are 1.0
        Assert.That(data, Is.All.EqualTo(1.0));
    }

    [Test]
    public void Zeros_ShouldCreateGpuVolumeFilledWithZeros()
    {
        // Arrange
        var shape = new Shape(3, 3); // Shape [3, 3, 1, 1]

        // Act
        var zerosGpuVolume = Volume<double>.Zeros(shape);

        // Assert
        Assert.That(zerosGpuVolume.Shape.Dimensions, Is.EqualTo(new[] { 3, 3, 1, 1 }));
        Assert.That(zerosGpuVolume.Shape.TotalLength, Is.EqualTo(9)); // Total length = 3 * 3 * 1 * 1

        var data = zerosGpuVolume.GetData();

        // Ensure all values are 0.0
        Assert.That(data, Is.All.EqualTo(0.0));
    }
}