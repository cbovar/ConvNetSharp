//using NUnit.Framework;

//namespace ConvNetSharp.V2.Tests;

//[TestFixture]
//public class CpuStorageTests
//{
//    [Test]
//    public void GetData_ShouldReturnCorrectData()
//    {
//        // Arrange
//        var data = new[] { 1.0, 2.0, 3.0 };
//        var shape = new Shape(1, 3);
//        var storage = new CpuStorage<double>(data, shape);

//        // Act
//        var result = storage.GetData();

//        // Assert
//        Assert.That(result, Is.EqualTo(data));
//    }

//    [Test]
//    public void ApplyOperation_ShouldModifyData()
//    {
//        // Arrange
//        var data = new[] { 1.0, 2.0, 3.0 };
//        var shape = new Shape(1, 3);
//        var storage = new CpuStorage<double>(data, shape);

//        // Act
//        storage.ApplyOperation(x => x * 2);

//        // Assert
//        Assert.That(storage.GetData(), Is.EqualTo((double[])[2.0, 4.0, 6.0]));
//    }

//    [Test]
//    public void Add_ShouldSumTwoStorages()
//    {
//        // Arrange
//        var data1 = new[] { 1.0, 2.0, 3.0 };
//        var data2 = new[] { 4.0, 5.0, 6.0 };
//        var resultData = new double[data1.Length];
//        var shape = new Shape(1, 3);

//        var storage1 = new CpuStorage<double>(data1, shape);
//        var storage2 = new CpuStorage<double>(data2, shape);
//        var resultStorage = new CpuStorage<double>(resultData, shape);

//        // Act
//        storage1.Add(storage2, resultStorage);

//        // Assert
//        Assert.That(resultStorage.GetData(), Is.EqualTo((double[])[5.0, 7.0, 9.0]));
//    }

//    [Test]
//    public void Add_ShouldThrowOnShapeMismatch()
//    {
//        // Arrange
//        var data1 = new[] { 1.0, 2.0, 3.0 };
//        var data2 = new[] { 4.0, 5.0 };
//        var resultData = new double[data1.Length];
//        var shape1 = new Shape(1, 3);
//        var shape2 = new Shape(1, 2);

//        var storage1 = new CpuStorage<double>(data1, shape1);
//        var storage2 = new CpuStorage<double>(data2, shape2);
//        var resultStorage = new CpuStorage<double>(resultData, shape1);

//        // Act & Assert
//        Assert.Throws<ArgumentException>(() => storage1.Add(storage2, resultStorage));
//    }

//    [Test]
//    public void Add_WithFloat_ShouldSumTwoStorages()
//    {
//        // Arrange
//        var data1 = new[] { 1.0f, 2.0f, 3.0f };
//        var data2 = new[] { 4.0f, 5.0f, 6.0f };
//        var resultData = new float[data1.Length];
//        var shape = new Shape(1, 3);

//        var storage1 = new CpuStorage<float>(data1, shape);
//        var storage2 = new CpuStorage<float>(data2, shape);
//        var resultStorage = new CpuStorage<float>(resultData, shape);

//        // Act
//        storage1.Add(storage2, resultStorage);

//        // Assert
//        Assert.That(resultStorage.GetData(), Is.EqualTo((float[])[5.0f, 7.0f, 9.0f]));
//    }

//    [Test]
//    public void Clear_ShouldResetAllValuesToZero()
//    {
//        // Arrange
//        var shape = new Shape(1, 3);
//        var data = new[] { 1.0, 2.0, 3.0 }; // Initial data
//        var storage = new CpuStorage<double>(data, shape);

//        // Act
//        storage.Clear();

//        // Assert
//        var clearedData = storage.GetData();
//        Assert.That(clearedData, Is.All.EqualTo(0.0), "All elements should be cleared to zero.");
//    }

//    [Test]
//    public void Add_ShouldAddTwoCpuStorages()
//    {
//        // Arrange
//        var shape = new Shape(1, 3);
//        var data1 = new[] { 1.0f, 2.0f, 3.0f };
//        var data2 = new[] { 4.0f, 5.0f, 6.0f };

//        var cpuStorage1 = new CpuStorage<float>(data1, shape);
//        var cpuStorage2 = new CpuStorage<float>(data2, shape);

//        // Create a result storage initialized with zeroes
//        var resultData = new[] { 0.0f, 0.0f, 0.0f };
//        var resultStorage = new CpuStorage<float>(resultData, shape);

//        // Act
//        cpuStorage1.Add(cpuStorage2, resultStorage);

//        // Assert
//        var result = resultStorage.GetData();
//        Assert.That(result, Is.EqualTo((float[])[5.0f, 7.0f, 9.0f]));
//    }

//    [Test]
//    public void Add_ShouldHandleCumulativeAddition()
//    {
//        // Arrange
//        var shape = new Shape(1, 3);
//        var data1 = new[] { 1.0f, 2.0f, 3.0f };
//        var data2 = new[] { 4.0f, 5.0f, 6.0f };
//        var data3 = new[] { 7.0f, 8.0f, 9.0f };

//        var cpuStorage1 = new CpuStorage<float>(data1, shape);
//        var cpuStorage2 = new CpuStorage<float>(data2, shape);
//        var cpuStorage3 = new CpuStorage<float>(data3, shape);

//        // Create a result storage initialized with zeroes
//        var resultData = new[] { 0.0f, 0.0f, 0.0f };
//        var resultStorage = new CpuStorage<float>(resultData, shape);

//        // Act
//        cpuStorage1.Add(cpuStorage2, resultStorage); // result = data1 + data2
//        cpuStorage3.Add(resultStorage, resultStorage); // result += data3

//        // Assert
//        var result = resultStorage.GetData();
//        Assert.That(result, Is.EqualTo((float[])[12.0f, 15.0f, 18.0f]));
//    }


//    [Test]
//    public void Add_ShouldThrowExceptionForShapeMismatch()
//    {
//        // Arrange
//        var shape1 = new Shape(1, 3);
//        var shape2 = new Shape(1, 4);

//        var data1 = new[] { 1.0f, 2.0f, 3.0f };
//        var data2 = new[] { 4.0f, 5.0f, 6.0f, 7.0f };

//        var cpuStorage1 = new CpuStorage<float>(data1, shape1);
//        var cpuStorage2 = new CpuStorage<float>(data2, shape2);

//        // Create a result storage with mismatched shape
//        var resultData = new[] { 0.0f, 0.0f, 0.0f };
//        var resultStorage = new CpuStorage<float>(resultData, shape1);

//        // Act & Assert
//        Assert.Throws<ArgumentException>(() => cpuStorage1.Add(cpuStorage2, resultStorage));
//    }


//    [Test]
//    public void Convolution_ShouldComputeCorrectOutput_For2DInput()
//    {
//        // Arrange
//        var inputData = Enumerable.Repeat(1.0f, 27).ToArray(); // Fill input with ones
//        var input = new CpuStorage<float>(inputData, new Shape(3, 3, 3, 1)); // 3x3x3x1

//        var filterData =
//            Enumerable.Repeat(1.0f, 12).Concat(Enumerable.Repeat(2.0f, 12)).ToArray(); // Filters with values 1 and 2
//        var filter = new CpuStorage<float>(filterData, new Shape(2, 2, 3, 2)); // 2x2x3x2

//        var result = new CpuStorage<float>(new float[2], new Shape(1, 1, 2, 1)); // Output 1x1x2x1

//        // Act
//        input.Convolution(filter, 0, 0, 2, result);

//        // Assert
//        Assert.That(result.Shape.Dimensions[0], Is.EqualTo(1));
//        Assert.That(result.Shape.Dimensions[1], Is.EqualTo(1));
//        Assert.That(result.Shape.Dimensions[2], Is.EqualTo(2));
//        Assert.That(result.Shape.Dimensions[3], Is.EqualTo(1));

//        Assert.That(result.Get(0, 0, 0, 0), Is.EqualTo(12.0f));
//        Assert.That(result.Get(0, 0, 1, 0), Is.EqualTo(24.0f));
//    }

//    [Test]
//    public void Convolution_ShouldComputeCorrectOutput_For1DInput()
//    {
//        // Arrange
//        var inputData = Enumerable.Repeat(1.0f, 9).ToArray(); // Fill input with ones
//        var input = new CpuStorage<float>(inputData, new Shape(9, 1, 1, 1)); // 9x1x1x1

//        var filterData = Enumerable.Repeat(1.0f, 6).ToArray(); // Filters with values 1
//        var filter = new CpuStorage<float>(filterData, new Shape(3, 1, 1, 2)); // 3x1x1x2

//        var result = new CpuStorage<float>(new float[18], new Shape(9, 1, 2, 1)); // Output 9x1x2x1

//        // Act
//        input.Convolution(filter, 1, 0, 1, result);

//        // Assert
//        Assert.That(result.Get(0, 0, 0, 0), Is.EqualTo(2.0f));
//        Assert.That(result.Get(0, 0, 1, 0), Is.EqualTo(2.0f));
//        Assert.That(result.Get(8, 0, 0, 0), Is.EqualTo(2.0f));
//        Assert.That(result.Get(8, 0, 1, 0), Is.EqualTo(2.0f));
//    }

//    [Test]
//    public void Convolution_ShouldComputeCorrectOutput_ForBatchInput()
//    {
//        // Arrange
//        var inputData = Enumerable.Repeat(1.0f, 27 * 2).ToArray(); // Fill input with ones
//        var input = new CpuStorage<float>(inputData, new Shape(3, 3, 3, 2)); // 3x3x3x2

//        var filterData =
//            Enumerable.Repeat(1.0f, 12).Concat(Enumerable.Repeat(2.0f, 12)).ToArray(); // Filters with values 1 and 2
//        var filter = new CpuStorage<float>(filterData, new Shape(2, 2, 3, 2)); // 2x2x3x2

//        var result = new CpuStorage<float>(new float[4], new Shape(1, 1, 2, 2)); // Output 1x1x2x2

//        // Act
//        input.Convolution(filter, 0, 0, 2, result);

//        // Assert
//        Assert.That(result.Shape.Dimensions[0], Is.EqualTo(1));
//        Assert.That(result.Shape.Dimensions[1], Is.EqualTo(1));
//        Assert.That(result.Shape.Dimensions[2], Is.EqualTo(2));
//        Assert.That(result.Shape.Dimensions[3], Is.EqualTo(2));

//        Assert.That(result.Get(0, 0, 0, 0), Is.EqualTo(12.0f));
//        Assert.That(result.Get(0, 0, 1, 0), Is.EqualTo(24.0f));
//        Assert.That(result.Get(0, 0, 0, 1), Is.EqualTo(12.0f));
//        Assert.That(result.Get(0, 0, 1, 1), Is.EqualTo(24.0f));
//    }

//    [Test]
//    public void Convolution_ShouldThrowOnShapeMismatch()
//    {
//        // Arrange
//        var inputData = Enumerable.Repeat(1.0f, 27).ToArray();
//        var input = new CpuStorage<float>(inputData, new Shape(3, 3, 3, 1)); // 3x3x3x1

//        // Use correct filter shape but mismatched data length
//        var invalidFilterData = Enumerable.Repeat(1.0f, 32).ToArray();
//        var invalidFilter = new CpuStorage<float>(invalidFilterData, new Shape(2, 2, 4, 2)); // 2x2x4x2

//        var result = new CpuStorage<float>(new float[2], new Shape(1, 1, 2, 1)); // Output 1x1x2x1

//        // Act & Assert
//        Assert.That(() => input.Convolution(invalidFilter, 0, 0, 2, result),
//            Throws.ArgumentException.With.Message.Contains("Input depth and filter depth must match."));
//    }
//}