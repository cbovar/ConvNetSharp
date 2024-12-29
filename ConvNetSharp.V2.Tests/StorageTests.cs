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
        var storage = CreateStorage(data, shape, device);

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
        var storage = CreateStorage(data, shape, device);

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

        var storage1 = CreateStorage(data1, shape, device);
        var storage2 = CreateStorage(data2, shape, device);
        var resultStorage = CreateStorage(resultData, shape, device);

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
        var storage1 = CreateStorage(data1, shape, device);
        var storage2 = CreateStorage(data2, shape, device);

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

        var storage1 = CreateStorage(data1, shape1, device);
        var storage2 = CreateStorage(data2, shape2, device);
        var resultStorage = CreateStorage(resultData, shape1, device);

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
        var storage = CreateStorage(data, shape, device);

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
        var input = CreateStorage(inputData, new Shape(3, 3, 3, 1), device);

        var filterData = Enumerable.Repeat(1.0f, 12).Concat(Enumerable.Repeat(2.0f, 12)).ToArray(); // Filters
        var filter = CreateStorage(filterData, new Shape(2, 2, 3, 2), device);

        var result = CreateStorage(new float[2], new Shape(1, 1, 2, 1), device);

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
        var storage = CreateStorage(data, shape, device);

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

        var storage1 = CreateStorage(data1, shape, device);
        var storage2 = CreateStorage(data2, shape, device);
        var resultStorage = CreateStorage(resultData, shape, device);

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

        var storage1 = CreateStorage(data1, shape, device);
        var storage2 = CreateStorage(data2, shape, device);
        var storage3 = CreateStorage(data3, shape, device);

        var resultData = new float[data1.Length];
        var resultStorage = CreateStorage(resultData, shape, device);

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
        var inputData = Enumerable.Repeat(1.0, 27).ToArray();
        var input = CreateStorage(inputData, new Shape(3, 3, 3, 1), device);

        var invalidFilterData = Enumerable.Repeat(1.0, 32).ToArray();
        var invalidFilter = CreateStorage(invalidFilterData, new Shape(2, 2, 4, 2), device);

        var result = CreateStorage(new double[2], new Shape(1, 1, 2, 1), device);

        Assert.Throws<ArgumentException>(() => input.Convolution(invalidFilter, 0, 0, 2, result));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Convolution_ShouldComputeCorrectOutput_For1DInput(Device device)
    {
        // Arrange
        var inputData = Enumerable.Repeat(1.0, 9).ToArray(); // Fill input with ones
        var input = CreateStorage(inputData, new Shape(9, 1, 1, 1), device);

        var filterData = Enumerable.Repeat(1.0, 6).ToArray(); // Filters with values 1
        var filter = CreateStorage(filterData, new Shape(3, 1, 1, 2), device);

        var result = CreateStorage(new double[18], new Shape(9, 1, 2, 1), device);

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
        var inputData = Enumerable.Repeat(1.0, 27 * 2).ToArray(); // Fill input with ones
        var input = CreateStorage(inputData, new Shape(3, 3, 3, 2), device);

        var filterData =
            Enumerable.Repeat(1.0, 12).Concat(Enumerable.Repeat(2.0, 12)).ToArray(); // Filters with values 1 and 2
        var filter = CreateStorage(filterData, new Shape(2, 2, 3, 2), device);

        var result = CreateStorage(new double[4], new Shape(1, 1, 2, 2), device);

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
    public void Convolution_ShouldThrowOnShapeMismatch_Float(Device device)
    {
        var inputData = Enumerable.Repeat(1.0f, 27).ToArray();
        var input = CreateStorage(inputData, new Shape(3, 3, 3, 1), device);

        var invalidFilterData = Enumerable.Repeat(1.0f, 32).ToArray();
        var invalidFilter = CreateStorage(invalidFilterData, new Shape(2, 2, 4, 2), device);

        var result = CreateStorage(new float[2], new Shape(1, 1, 2, 1), device);

        Assert.Throws<ArgumentException>(() => input.Convolution(invalidFilter, 0, 0, 2, result));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Convolution_ShouldComputeCorrectOutput_For1DInput_Float(Device device)
    {
        // Arrange
        var inputData = Enumerable.Repeat(1.0f, 9).ToArray(); // Fill input with ones
        var input = CreateStorage(inputData, new Shape(9, 1, 1, 1), device);

        var filterData = Enumerable.Repeat(1.0f, 6).ToArray(); // Filters with values 1
        var filter = CreateStorage(filterData, new Shape(3, 1, 1, 2), device);

        var result = CreateStorage(new float[18], new Shape(9, 1, 2, 1), device);

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
    public void Convolution_ShouldComputeCorrectOutput_ForBatchInput_Float(Device device)
    {
        // Arrange
        var inputData = Enumerable.Repeat(1.0f, 27 * 2).ToArray(); // Fill input with ones
        var input = CreateStorage(inputData, new Shape(3, 3, 3, 2), device);

        var filterData =
            Enumerable.Repeat(1.0f, 12).Concat(Enumerable.Repeat(2.0f, 12)).ToArray(); // Filters with values 1 and 2
        var filter = CreateStorage(filterData, new Shape(2, 2, 3, 2), device);

        var result = CreateStorage(new float[4], new Shape(1, 1, 2, 2), device);

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
        var storage = CreateStorage(data, shape, device);

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
        var storage = CreateStorage(data, shape, device);

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
        var storage = CreateStorage(data, shape, device);

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

        var storage = CreateStorage(data, shape, device);

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

        var storage = CreateStorage(data, shape, device);

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
        var storage = CreateStorage(data, shape, device);

        var coordinates = new[] { 2, 3, 1, 1 }; // Coordinates: (w=2, h=3, c=1, n=1)
        var expectedValue = 42.0;

        // Act
        storage.Set(coordinates, expectedValue);

        // Assert: Verify the value using the storage abstraction
        var actualValue = storage.Get(coordinates);
        Assert.That(actualValue, Is.EqualTo(expectedValue));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void MultiplyByScalar_ShouldScaleAllElements(Device device)
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0 };
        var shape = new Shape(2, 2);
        var storage = CreateStorage(data, shape, device);
        var scalar = 3.0;
        var expectedData = new[] { 3.0, 6.0, 9.0, 12.0 };

        // Create result storage
        var resultStorage = CreateStorage(new double[data.Length], shape, device);

        // Act
        storage.MultiplyByScalar(scalar, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
        Assert.That(storage.GetData(), Is.EqualTo(data)); // Original storage remains unchanged
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void MultiplyByScalar_ShouldScaleInPlace(Device device)
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0 };
        var shape = new Shape(2, 2);
        var storage = CreateStorage(data, shape, device);
        var scalar = 3.0;
        var expectedData = new[] { 3.0, 6.0, 9.0, 12.0 };

        // Act (in-place)
        storage.MultiplyByScalar(scalar, storage);

        // Assert
        Assert.That(storage.GetData(), Is.EqualTo(expectedData)); // Original storage is modified
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void MultiplyByScalar_ShouldScaleAllElements_Float(Device device)
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var shape = new Shape(2, 2);
        var storage = CreateStorage(data, shape, device);
        var scalar = 3.0f;
        var expectedData = new[] { 3.0f, 6.0f, 9.0f, 12.0f };

        // Create result storage
        var resultStorage = CreateStorage(new float[data.Length], shape, device);

        // Act
        storage.MultiplyByScalar(scalar, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
        Assert.That(storage.GetData(), Is.EqualTo(data)); // Original storage remains unchanged
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void MultiplyByScalar_ShouldScaleInPlace_Float(Device device)
    {
        // Arrange
        var data = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var shape = new Shape(2, 2);
        var storage = CreateStorage(data, shape, device);
        var scalar = 3.0f;
        var expectedData = new[] { 3.0f, 6.0f, 9.0f, 12.0f };

        // Act (in-place)
        storage.MultiplyByScalar(scalar, storage);

        // Assert
        Assert.That(storage.GetData(), Is.EqualTo(expectedData)); // Original storage is modified
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Subtract_ShouldComputeCorrectResult(Device device)
    {
        // Arrange
        var data1 = new[] { 5.0, 6.0, 7.0 };
        var data2 = new[] { 3.0, 2.0, 1.0 };
        var shape = new Shape(1, 3);

        var storage1 = CreateStorage(data1, shape, device);
        var storage2 = CreateStorage(data2, shape, device);
        var resultStorage = CreateStorage(new double[data1.Length], shape, device);

        // Act
        storage1.Subtract(storage2, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(new[] { 2.0, 4.0, 6.0 }));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Subtract_ShouldHandleBroadcasting(Device device)
    {
        // Arrange
        var leftData = new[]
        {
            5.0, 6.0,
            7.0, 8.0,
            9.0, 10.0
        }; // 2x3x1x1 matrix
        var rightData = new[]
        {
            2.0,
            1.0,
            3.0
        }; // 1x3x1x1 (broadcast across rows)
        var expectedData = new[]
        {
            3.0, 4.0,
            6.0, 7.0,
            6.0, 7.0
        }; // 2x3x1x1 Result of broadcasting and subtraction

        var leftShape = new Shape(2, 3); // Shape: 2 rows, 3 columns
        var rightShape = new Shape(1, 3); // Shape: 1 row, 3 columns (broadcastable)
        var resultShape = new Shape(2, 3); // Broadcasted shape: 2 rows, 3 columns

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new double[resultShape.TotalLength], resultShape, device);

        // Act
        leftStorage.Subtract(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Subtract_ShouldHandleBroadcasting2(Device device)
    {
        // Arrange
        var leftData = new[] { 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 }; // 2x3 matrix
        var rightData = new[] { 1.0 }; // 1x1 (broadcast across rows)
        var expectedData = new[] { 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 }; // Result of broadcasting and subtraction

        var leftShape = new Shape(2, 3); // Shape: 2 rows, 3 columns
        var rightShape = new Shape(1, 1); // Shape: 1 row, 1 columns (broadcastable)
        var resultShape = new Shape(2, 3); // Broadcasted shape: 2 rows, 3 columns

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new double[resultShape.TotalLength], resultShape, device);

        // Act
        leftStorage.Subtract(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }


    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void MatMultiply_ShouldComputeCorrectResult_3x3(Device device)
    {
        // Arrange
        var leftData = new[]
        {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        }; // 3x3 matrix
        var rightData = new[]
        {
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0
        }; // 3x3 matrix
        var expectedResult = new[]
        {
            30.0, 24.0, 18.0,
            84.0, 69.0, 54.0,
            138.0, 114.0, 90.0
        }; // 3x3 result matrix

        // Shape reflects (columns, rows, channels, batch)
        var leftShape = new Shape(3, 3, 1, 1); // 3 rows, 3 columns
        var rightShape = new Shape(3, 3, 1, 1); // 3 rows, 3 columns
        var resultShape = new Shape(3, 3, 1, 1); // Result: 3 rows, 3 columns

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new double[9], resultShape, device);

        // Act
        leftStorage.MatMultiply(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedResult));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void MatMultiply_NonSquareMatrices_ShouldComputeCorrectResult(Device device)
    {
        // Arrange
        var leftData = new[]
        {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        }; // 2x4 matrix (2 rows, 4 columns)

        var rightData = new[]
        {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        }; // 3x2 matrix (3 rows, 2 columns)

        var expectedResult = new[]
        {
            9.0, 12.0, 15.0,
            19.0, 26.0, 33.0,
            29.0, 40.0, 51.0,
            39.0, 54.0, 69.0
        }; // 3x4 result matrix (3 rows, 4 columns)

        // Define the shapes
        var leftShape = new Shape(2, 4, 1, 1); // 2 rows, 4 columns
        var rightShape = new Shape(3, 2, 1, 1); // 3 rows, 2 columns
        var resultShape = new Shape(3, 4, 1, 1); // Result: 3 rows, 4 columns

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new double[12], resultShape, device);

        // Act
        leftStorage.MatMultiply(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedResult));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void MatMultiply_ShouldComputeCorrectResult_3x3_Float(Device device)
    {
        // Arrange
        var leftData = new[]
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f
        }; // 3x3 matrix
        var rightData = new[]
        {
            9.0f, 8.0f, 7.0f,
            6.0f, 5.0f, 4.0f,
            3.0f, 2.0f, 1.0f
        }; // 3x3 matrix
        var expectedResult = new[]
        {
            30.0f, 24.0f, 18.0f,
            84.0f, 69.0f, 54.0f,
            138.0f, 114.0f, 90.0f
        }; // 3x3 result matrix

        // Shape reflects (columns, rows, channels, batch)
        var leftShape = new Shape(3, 3, 1, 1); // 3 rows, 3 columns
        var rightShape = new Shape(3, 3, 1, 1); // 3 rows, 3 columns
        var resultShape = new Shape(3, 3, 1, 1); // Result: 3 rows, 3 columns

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new float[9], resultShape, device);

        // Act
        leftStorage.MatMultiply(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedResult));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void MatMultiply_NonSquareMatrices_ShouldComputeCorrectResult_Float(Device device)
    {
        // Arrange
        var leftData = new[]
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f
        }; // 2x4 matrix (2 rows, 4 columns)

        var rightData = new[]
        {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f
        }; // 3x2 matrix (3 rows, 2 columns)

        var expectedResult = new[]
        {
            9.0f, 12.0f, 15.0f,
            19.0f, 26.0f, 33.0f,
            29.0f, 40.0f, 51.0f,
            39.0f, 54.0f, 69.0f
        }; // 3x4 result matrix (3 rows, 4 columns)

        // Define the shapes
        var leftShape = new Shape(2, 4, 1, 1); // 2 rows, 4 columns
        var rightShape = new Shape(3, 2, 1, 1); // 3 rows, 2 columns
        var resultShape = new Shape(3, 4, 1, 1); // Result: 3 rows, 4 columns

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new float[12], resultShape, device);

        // Act
        leftStorage.MatMultiply(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedResult));
    }


    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Multiply_ShouldElementWiseMultiplyTwoStorages(Device device)
    {
        // Arrange
        var data1 = new[] { 1.0, 2.0, 3.0 };
        var data2 = new[] { 4.0, 5.0, 6.0 };
        var resultData = new double[data1.Length];
        var shape = new Shape(1, 3);

        var storage1 = CreateStorage(data1, shape, device);
        var storage2 = CreateStorage(data2, shape, device);
        var resultStorage = CreateStorage(resultData, shape, device);

        // Act
        storage1.Multiply(storage2, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(new[] { 4.0, 10.0, 18.0 }));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void SetData_ShouldUpdateStorage(Device device)
    {
        // Arrange
        var initialData = new[] { 1.0, 2.0, 3.0 };
        var newData = new[] { 4.0, 5.0, 6.0 };
        var shape = new Shape(1, 3);

        var storage = CreateStorage(initialData, shape, device);

        // Act
        storage.SetData(newData);

        // Assert
        Assert.That(storage.GetData(), Is.EqualTo(newData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Transpose_ShouldSwapRowsAndColumns(Device device)
    {
        // Arrange
        var inputData = new[]
        {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        }; // 2x3 matrix
        var expectedTransposedData = new[]
        {
            1.0, 4.0,
            2.0, 5.0,
            3.0, 6.0
        }; // 3x2 matrix after transpose

        var inputShape = new Shape(3, 2); // Columns x Rows
        var resultShape = new Shape(2, 3); // Rows x Columns after transpose

        var inputStorage = CreateStorage(inputData, inputShape, device);
        var resultStorage = CreateStorage(new double[resultShape.TotalLength], resultShape, device);

        // Act
        inputStorage.Transpose(resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedTransposedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Transpose_ShouldThrowOnShapeMismatch(Device device)
    {
        // Arrange
        var inputData = new[]
        {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        }; // 2x3 matrix
        var inputShape = new Shape(3, 2); // Columns x Rows

        // Mismatched result shape
        var mismatchedShape = new Shape(4, 4);
        var inputStorage = CreateStorage(inputData, inputShape, device);
        var resultStorage = CreateStorage(new double[mismatchedShape.TotalLength], mismatchedShape, device);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => inputStorage.Transpose(resultStorage));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Transpose_ShouldSwapRowsAndColumns_Float(Device device)
    {
        // Arrange
        var inputData = new[]
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        }; // 2x3 matrix
        var expectedTransposedData = new[]
        {
            1.0f, 4.0f,
            2.0f, 5.0f,
            3.0f, 6.0f
        }; // 3x2 matrix after transpose

        var inputShape = new Shape(3, 2); // Columns x Rows
        var resultShape = new Shape(2, 3); // Rows x Columns after transpose

        var inputStorage = CreateStorage(inputData, inputShape, device);
        var resultStorage = CreateStorage(new float[resultShape.TotalLength], resultShape, device);

        // Act
        inputStorage.Transpose(resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedTransposedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Transpose_ShouldThrowOnShapeMismatch_Float(Device device)
    {
        // Arrange
        var inputData = new[]
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        }; // 2x3 matrix
        var inputShape = new Shape(3, 2); // Columns x Rows

        // Mismatched result shape
        var mismatchedShape = new Shape(4, 4);
        var inputStorage = CreateStorage(inputData, inputShape, device);
        var resultStorage = CreateStorage(new float[mismatchedShape.TotalLength], mismatchedShape, device);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => inputStorage.Transpose(resultStorage));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Power_ShouldRaiseToThePowerOfEachElement(Device device)
    {
        // Arrange
        var baseData = new[] { 1.0, 2.0, 3.0, 4.0 };
        var exponentData = new[] { 0.0, 1.0, 2.0, 3.0 };
        var expectedData = new[] { 1.0, 2.0, 9.0, 64.0 }; // 1^0, 2^1, 3^2, 4^3
        var shape = new Shape(2, 2);

        var baseStorage = CreateStorage(baseData, shape, device);
        var exponentStorage = CreateStorage(exponentData, shape, device);
        var resultStorage = CreateStorage(new double[baseData.Length], shape, device);

        // Act
        baseStorage.Power(exponentStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Power_ShouldRaiseToThePowerOfEachElement_Float(Device device)
    {
        // Arrange
        var baseData = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var exponentData = new[] { 0.0f, 1.0f, 2.0f, 3.0f };
        var expectedData = new[] { 1.0f, 2.0f, 9.0f, 64.0f }; // 1^0, 2^1, 3^2, 4^3
        var shape = new Shape(2, 2);

        var baseStorage = CreateStorage(baseData, shape, device);
        var exponentStorage = CreateStorage(exponentData, shape, device);
        var resultStorage = CreateStorage(new float[baseData.Length], shape, device);

        // Act
        baseStorage.Power(exponentStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Power_ShouldHandleInPlaceOperation(Device device)
    {
        // Arrange
        var baseData = new[] { 1.0, 2.0, 3.0, 4.0 };
        var exponentData = new[] { 0.0, 1.0, 2.0, 3.0 };
        var expectedData = new[] { 1.0, 2.0, 9.0, 64.0 }; // 1^0, 2^1, 3^2, 4^3
        var shape = new Shape(2, 2);

        var baseStorage = CreateStorage(baseData, shape, device);
        var exponentStorage = CreateStorage(exponentData, shape, device);

        // Act (in-place)
        baseStorage.Power(exponentStorage, baseStorage);

        // Assert
        Assert.That(baseStorage.GetData(), Is.EqualTo(expectedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Power_ShouldThrowOnShapeMismatch(Device device)
    {
        // Arrange
        var baseData = new[] { 1.0, 2.0, 3.0 };
        var exponentData = new[] { 2.0, 3.0 };
        var shape1 = new Shape(1, 3);
        var shape2 = new Shape(1, 2);

        var baseStorage = CreateStorage(baseData, shape1, device);
        var exponentStorage = CreateStorage(exponentData, shape2, device);
        var resultStorage = CreateStorage(new double[baseData.Length], shape1, device);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => baseStorage.Power(exponentStorage, resultStorage));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Power_ShouldHandleBroadcasting(Device device)
    {
        // Arrange
        var baseData = new[] { 1.0, 2.0, 3.0, 4.0 };
        var exponentData = new[] { 2.0 }; // Scalar exponent for broadcasting
        var expectedData = new[] { 1.0, 4.0, 9.0, 16.0 }; // 1^2, 2^2, 3^2, 4^2
        var shape = new Shape(2, 2);

        var baseStorage = CreateStorage(baseData, shape, device);
        var exponentStorage = CreateStorage(exponentData, new Shape(1), device);
        var resultStorage = CreateStorage(new double[baseData.Length], shape, device);

        // Act
        baseStorage.Power(exponentStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Add_ShouldHandleBroadcasting(Device device)
    {
        // Arrange
        var leftData = new[] { 1.0, 2.0, 3.0 };
        var rightData = new[] { 1.0 }; // Scalar broadcasting
        var expectedData = new[] { 2.0, 3.0, 4.0 };
        var shape = new Shape(1, 3);

        var leftStorage = CreateStorage(leftData, shape, device);
        var rightStorage = CreateStorage(rightData, new Shape(1), device);
        var resultStorage = CreateStorage(new double[3], shape, device);

        // Act
        leftStorage.Add(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }


    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Multiply_ShouldHandleBroadcasting(Device device)
    {
        // Arrange
        var leftData = new[] { 1.0, 2.0, 3.0, 4.0 }; // Shape: 2x2
        var rightData = new[] { 2.0 }; // Shape: 1x1 (broadcasted scalar)
        var expectedData = new[] { 2.0, 4.0, 6.0, 8.0 }; // Result of broadcasting and multiplication

        var leftShape = new Shape(2, 2);
        var rightShape = new Shape(1, 1);
        var resultShape = new Shape(2, 2);

        Console.WriteLine($"Left Shape: {leftShape}");
        Console.WriteLine($"Right Shape: {rightShape}");
        Console.WriteLine($"Result Shape: {resultShape}");

        // Correct validation direction
        Assert.That(rightShape.IsBroadcastableTo(leftShape), Is.True, "Right should broadcast to Left.");
        Assert.That(leftShape.IsBroadcastableTo(resultShape), Is.True, "Left should broadcast to Result.");

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new double[resultShape.TotalLength], resultShape, device);

        // Act
        leftStorage.Multiply(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Multiply_ShouldBroadcastAcrossRows(Device device)
    {
        // Arrange
        var leftData = new[] { 1.0, 2.0, 
                               3.0, 4.0, 
                               5.0, 6.0 }; // Shape: 2x3
        var rightData = new[] { 1.0, 
                                2.0, 
                                3.0 }; // Shape: 1x3 (broadcast across rows)
        var expectedData = new[] { 1.0, 2.0,
                                   6.0, 8.0, 
                                   15.0, 18.0 }; // Result of broadcasting and multiplication

        var leftShape = new Shape(2, 3);
        var rightShape = new Shape(1, 3);
        var resultShape = new Shape(2, 3);

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new double[resultShape.TotalLength], resultShape, device);

        // Act
        leftStorage.Multiply(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }

    [TestCase(Device.CPU)]
    [TestCase(Device.GPU)]
    public void Multiply_ShouldHandleScalarBroadcasting(Device device)
    {
        // Arrange
        var leftData = new[] { 1.0, 2.0, 3.0, 4.0 }; // Shape: 2x2
        var rightData = new[] { 2.0 }; // Scalar
        var expectedData = new[] { 2.0, 4.0, 6.0, 8.0 }; // Each element multiplied by scalar

        var leftShape = new Shape(2, 2);
        var rightShape = new Shape(1);
        var resultShape = new Shape(2, 2);

        var leftStorage = CreateStorage(leftData, leftShape, device);
        var rightStorage = CreateStorage(rightData, rightShape, device);
        var resultStorage = CreateStorage(new double[resultShape.TotalLength], resultShape, device);

        // Act
        leftStorage.Multiply(rightStorage, resultStorage);

        // Assert
        Assert.That(resultStorage.GetData(), Is.EqualTo(expectedData));
    }


    private static IVolumeStorage<T> CreateStorage<T>(T[] data, Shape shape, Device device) where T : struct, INumber<T>
    {
        return device switch
        {
            Device.CPU => new CpuStorage<T>(data, shape),
            Device.GPU => new GpuStorage<T>(data, shape),
            _ => throw new ArgumentException("Unsupported device type.")
        };
    }
}