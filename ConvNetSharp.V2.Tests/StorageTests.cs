using System.Numerics;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class StorageTests
{
    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void GetData_ShouldReturnCorrectData(Device device)
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var storage = this.CreateStorage(data, shape, device);

        // Act
        var result = storage.GetData();

        // Assert
        Assert.That(result, Is.EqualTo(data));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void ApplyOperation_ShouldModifyData(Device device)
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0 };
        var shape = new Shape(1, 3);
        var storage = this.CreateStorage(data, shape, device);

        // Act
        storage.ApplyOperation(x => x * 2);

        // Assert
        Assert.That(storage.GetData(), Is.EqualTo(new[] { 2.0, 4.0, 6.0 }));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Add_ShouldSumTwoStorages(Device device)
    {
        // Arrange
        var data1 = new[] { 1.0, 2.0, 3.0 };
        var data2 = new[] { 4.0, 5.0, 6.0 };
        var resultData = new double[data1.Length];
        var shape = new Shape(1, 3);

        var storage1 = this.CreateStorage(data1, shape, device);
        var storage2 = this.CreateStorage(data2, shape, device);
        var resultStorage = this.CreateStorage(resultData, shape, device);

        // Act
        storage1.Add(storage2, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(new[] { 5.0, 7.0, 9.0 }));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Add_ShouldAccumulateInPlace(Device device)
    {
        // Arrange
        var shape = new Shape(1, 3);
        var data1 = new[] { 1.0, 2.0, 3.0 };
        var data2 = new[] { 4.0, 5.0, 6.0 };

        // Create the storage with the same data
        var storage1 = this.CreateStorage(data1, shape, device);
        var storage2 = this.CreateStorage(data2, shape, device);

        // Log initial states
        Console.WriteLine($"Initial storage1: {string.Join(", ", storage1.GetData())}");
        Console.WriteLine($"Initial storage2: {string.Join(", ", storage2.GetData())}");

        // Act: Accumulate data from storage1 into storage2
        storage1.Add(storage2);

        // Log final state
        Console.WriteLine($"After storage1.Add(storage2): {string.Join(", ", storage2.GetData())}");

        // Assert: Ensure storage2 is updated in place
        Assert.That(storage2.GetData(), Is.EqualTo(new[] { 5.0, 7.0, 9.0 }));
    }


    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Add_ShouldThrowOnShapeMismatch(Device device)
    {
        // Arrange
        var data1 = new[] { 1.0, 2.0, 3.0 };
        var data2 = new[] { 4.0, 5.0 };
        var resultData = new double[data1.Length];
        var shape1 = new Shape(1, 3);
        var shape2 = new Shape(1, 2);

        var storage1 = this.CreateStorage(data1, shape1, device);
        var storage2 = this.CreateStorage(data2, shape2, device);
        var resultStorage = this.CreateStorage(resultData, shape1, device);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => storage1.Add(storage2, resultStorage));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Clear_ShouldResetAllValuesToZero(Device device)
    {
        // Arrange
        var shape = new Shape(1, 3);
        var data = new[] { 1.0, 2.0, 3.0 };
        var storage = this.CreateStorage(data, shape, device);

        // Act
        storage.Clear();

        // Assert
        var clearedData = storage.GetData();
        Assert.That(clearedData, Is.All.EqualTo(0.0));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Convolution_ShouldComputeCorrectOutput_For2DInput(Device device)
    {
        // Arrange
        var inputData = Enumerable.Repeat(1.0f, 27).ToArray(); // Fill input with ones
        var input = this.CreateStorage(inputData, new Shape(3, 3, 3, 1), device);

        var filterData = Enumerable.Repeat(1.0f, 12).Concat(Enumerable.Repeat(2.0f, 12)).ToArray(); // Filters
        var filter = this.CreateStorage(filterData, new Shape(2, 2, 3, 2), device);

        var result = this.CreateStorage(new float[2], new Shape(1, 1, 2, 1), device);

        // Act
        input.Convolution(filter, 0, 0, 2, result);

        // Assert
        Assert.That(result.Shape.Dimensions[0], Is.EqualTo(1));
        Assert.That(result.Shape.Dimensions[1], Is.EqualTo(1));
        Assert.That(result.Shape.Dimensions[2], Is.EqualTo(2));
        Assert.That(result.Shape.Dimensions[3], Is.EqualTo(1));

        Assert.That(result.Get(0, 0, 0, 0), Is.EqualTo(12.0f));
        Assert.That(result.Get(0, 0, 1, 0), Is.EqualTo(24.0f));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Get_ShouldThrowIndexOutOfBounds(Device device)
    {
        // Arrange
        var data = new double[5];
        var shape = new Shape(5);
        var storage = this.CreateStorage(data, shape, device);

        // Act & Assert
        Assert.Throws<IndexOutOfRangeException>(() => storage.Get(10));
    }


    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Add_WithFloat_ShouldSumTwoStorages(Device device)
    {
        var data1 = new[] { 1.0f, 2.0f, 3.0f };
        var data2 = new[] { 4.0f, 5.0f, 6.0f };
        var resultData = new float[data1.Length];
        var shape = new Shape(1, 3);

        var storage1 = this.CreateStorage(data1, shape, device);
        var storage2 = this.CreateStorage(data2, shape, device);
        var resultStorage = this.CreateStorage(resultData, shape, device);

        storage1.Add(storage2, resultStorage);

        Assert.That(resultStorage.GetData(), Is.EqualTo(new[] { 5.0f, 7.0f, 9.0f }));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Add_ShouldHandleCumulativeAddition_WithLogs(Device device)
    {
        var shape = new Shape(1, 3);
        var data1 = new[] { 1.0f, 2.0f, 3.0f };
        var data2 = new[] { 4.0f, 5.0f, 6.0f };
        var data3 = new[] { 7.0f, 8.0f, 9.0f };

        var storage1 = this.CreateStorage(data1, shape, device);
        var storage2 = this.CreateStorage(data2, shape, device);
        var storage3 = this.CreateStorage(data3, shape, device);

        var resultData = new float[data1.Length];
        var resultStorage = this.CreateStorage(resultData, shape, device);

        // Log initial states
        Console.WriteLine($"Initial data1: {string.Join(", ", storage1.GetData())}");
        Console.WriteLine($"Initial data2: {string.Join(", ", storage2.GetData())}");
        Console.WriteLine($"Initial data3: {string.Join(", ", storage3.GetData())}");
        Console.WriteLine($"Initial resultStorage: {string.Join(", ", resultStorage.GetData())}");

        // Act: First addition
        storage1.Add(storage2, resultStorage);
        Console.WriteLine($"After storage1.Add(storage2): {string.Join(", ", resultStorage.GetData())}");

        // Second addition
        storage3.Add(resultStorage, resultStorage);
        Console.WriteLine($"After storage3.Add(resultStorage): {string.Join(", ", resultStorage.GetData())}");

        // Assert final state
        Assert.That(resultStorage.GetData(), Is.EqualTo(new[] { 12.0f, 15.0f, 18.0f }));
    }


    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Convolution_ShouldThrowOnShapeMismatch(Device device)
    {
        var inputData = Enumerable.Repeat(1.0f, 27).ToArray();
        var input = this.CreateStorage(inputData, new Shape(3, 3, 3, 1), device);

        var invalidFilterData = Enumerable.Repeat(1.0f, 32).ToArray();
        var invalidFilter = this.CreateStorage(invalidFilterData, new Shape(2, 2, 4, 2), device);

        var result = this.CreateStorage(new float[2], new Shape(1, 1, 2, 1), device);

        Assert.Throws<ArgumentException>(() => input.Convolution(invalidFilter, 0, 0, 2, result));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Convolution_ShouldComputeCorrectOutput_For1DInput(Device device)
    {
        // Arrange
        var inputData = Enumerable.Repeat(1.0f, 9).ToArray(); // Fill input with ones
        var input = this.CreateStorage(inputData, new Shape(9, 1, 1, 1), device);

        var filterData = Enumerable.Repeat(1.0f, 6).ToArray(); // Filters with values 1
        var filter = this.CreateStorage(filterData, new Shape(3, 1, 1, 2), device);

        var result = this.CreateStorage(new float[18], new Shape(9, 1, 2, 1), device);

        // Act
        input.Convolution(filter, 1, 0, 1, result);

        // Assert
        Assert.That(result.Get(0, 0, 0, 0), Is.EqualTo(2.0f));
        Assert.That(result.Get(0, 0, 1, 0), Is.EqualTo(2.0f));
        Assert.That(result.Get(8, 0, 0, 0), Is.EqualTo(2.0f));
        Assert.That(result.Get(8, 0, 1, 0), Is.EqualTo(2.0f));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Convolution_ShouldComputeCorrectOutput_ForBatchInput(Device device)
    {
        // Arrange
        var inputData = Enumerable.Repeat(1.0f, 27 * 2).ToArray(); // Fill input with ones
        var input = this.CreateStorage(inputData, new Shape(3, 3, 3, 2), device);

        var filterData =
            Enumerable.Repeat(1.0f, 12).Concat(Enumerable.Repeat(2.0f, 12)).ToArray(); // Filters with values 1 and 2
        var filter = this.CreateStorage(filterData, new Shape(2, 2, 3, 2), device);

        var result = this.CreateStorage(new float[4], new Shape(1, 1, 2, 2), device);

        // Act
        input.Convolution(filter, 0, 0, 2, result);

        // Assert
        Assert.That(result.Shape.Dimensions[0], Is.EqualTo(1));
        Assert.That(result.Shape.Dimensions[1], Is.EqualTo(1));
        Assert.That(result.Shape.Dimensions[2], Is.EqualTo(2));
        Assert.That(result.Shape.Dimensions[3], Is.EqualTo(2));

        Assert.That(result.Get(0, 0, 0, 0), Is.EqualTo(12.0f));
        Assert.That(result.Get(0, 0, 1, 0), Is.EqualTo(24.0f));
        Assert.That(result.Get(0, 0, 0, 1), Is.EqualTo(12.0f));
        Assert.That(result.Get(0, 0, 1, 1), Is.EqualTo(24.0f));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Set_ShouldSetValueAt2DCoordinates(Device device)
    {
        // Arrange
        var shape = new Shape(4, 4); // 4x4
        var data = new double[shape.TotalLength];
        var storage = this.CreateStorage(data, shape, device);

        // Act
        storage.Set(2, 3, 42.0);

        // Assert
        Assert.That(storage.Get(2, 3), Is.EqualTo(42.0));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Set_ShouldSetValueAt3DCoordinates(Device device)
    {
        // Arrange
        var shape = new Shape(4, 4, 3); // 4x4x3
        var data = new double[shape.TotalLength];
        var storage = this.CreateStorage(data, shape, device);

        // Act
        storage.Set(2, 3, 1, 42.0);

        // Assert
        Assert.That(storage.Get(2, 3, 1), Is.EqualTo(42.0));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Get_ShouldRetrieveValueAt2DCoordinates(Device device)
    {
        // Arrange
        var shape = new Shape(4, 4); // 4x4
        var data = new double[shape.TotalLength];
        data[14] = 42.0; // Pre-set value for coordinates (2, 3)
        var storage = this.CreateStorage(data, shape, device);

        // Act
        var value = storage.Get(2, 3);

        // Assert
        Assert.That(value, Is.EqualTo(42.0));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Get_ShouldRetrieveValueAt3DCoordinates(Device device)
    {
        // Arrange
        var shape = new Shape(4, 4, 3); // 4x4x3
        var data = new double[shape.TotalLength];

        // Calculate the flattened index for coordinates (2, 3, 1)
        var index = 1 * shape.Dimensions[0] * shape.Dimensions[1] + // Channel offset
                    3 * shape.Dimensions[0] + // Row offset
                    2; // Column offset
        data[index] = 42.0; // Pre-set value at the calculated index

        var storage = this.CreateStorage(data, shape, device);

        // Act
        var value = storage.Get(2, 3, 1); // Retrieve value at (2, 3, 1)

        // Assert
        Assert.That(value, Is.EqualTo(42.0));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Get_ShouldRetrieveValueAt4DCoordinates(Device device)
    {
        // Arrange
        var shape = new Shape(4, 4, 3, 2); // 4x4x3x2 (width, height, channel, batch)
        var data = new double[shape.TotalLength];

        // Correctly calculate the flattened index for coordinates (2, 3, 1, 1)
        var index = 1 * shape.Dimensions[0] * shape.Dimensions[1] * shape.Dimensions[2] + // Batch offset
                    1 * shape.Dimensions[0] * shape.Dimensions[1] + // Channel offset
                    3 * shape.Dimensions[0] + // Row offset
                    2; // Column offset
        data[index] = 42.0; // Pre-set value at the calculated index

        var storage = this.CreateStorage(data, shape, device);

        // Act
        var value = storage.Get(2, 3, 1, 1); // Retrieve value at (w=2, h=3, c=1, n=1)

        // Assert
        Assert.That(value, Is.EqualTo(42.0));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)] 
    public void Set_ShouldStoreValueAtNDimensionalCoordinates(Device device)
    {
        // Arrange
        var shape = new Shape(4, 4, 3, 2); // Shape is 4x4x3x2 (width, height, channel, batch)
        var data = new double[shape.TotalLength];
        var storage = this.CreateStorage(data, shape, device);

        var coordinates = new[] { 2, 3, 1, 1 }; // Coordinates: (w=2, h=3, c=1, n=1)
        var expectedValue = 42.0;

        // Act
        storage.Set(coordinates, expectedValue);

        // Assert: Verify the value using the storage abstraction
        var actualValue = storage.Get(coordinates);
        Assert.That(actualValue, Is.EqualTo(expectedValue));
    }

    private IVolumeStorage<T> CreateStorage<T>(T[] data, Shape shape, Device device) where T : struct, INumber<T>
    {
        return device switch
        {
            Device.CPU => new CpuStorage<T>(data, shape),
            Device.GPU => new GpuStorage<T>(data, shape),
            _ => throw new ArgumentException("Unsupported device type.")
        };
    }
}