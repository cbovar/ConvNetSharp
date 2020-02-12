using System.Linq;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using NUnit.Framework;

namespace ConvNetSharp.Core.Tests
{
    [TestFixture]
    public class FullyConnLayerTests
    {
        [Test]
        public void ComputeTwiceGradientShouldYieldTheSameResult()
        {
            const int inputWidth = 10;
            const int inputHeight = 10;
            const int inputDepth = 5;

            // Create layer
            var layer = new FullyConnLayer<double>(5) { BiasPref = 0.1 };
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = BuilderInstance<double>.Volume.Random(new Shape(inputWidth, inputHeight, inputDepth));
            var output = layer.DoForward(input, true);

            // Set output gradients to 1
            var outputGradient = BuilderInstance<double>.Volume.From(new double[output.Shape.TotalLength].Populate(1.0), output.Shape);

            // Backward pass to retrieve gradients
            layer.Backward(outputGradient);
            var step1 = layer.InputActivationGradients.Clone().ToArray();

            layer.Backward(outputGradient);
            var step2 = layer.InputActivationGradients.Clone().ToArray();

            Assert.IsTrue(step1.SequenceEqual(step2));
        }

        [Test]
        public void Forward()
        {
            const int inputWidth = 2;
            const int inputHeight = 2;
            const int inputDepth = 2;
            const int inputBatchSize = 2;

            var layer = new FullyConnLayer<double>(2) { BiasPref = 0.1 };
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Make sure filter shape had flatten input shape
            Assert.AreEqual(1, layer.Filters.Shape.Dimensions[0]);
            Assert.AreEqual(1, layer.Filters.Shape.Dimensions[1]);
            Assert.AreEqual(8, layer.Filters.Shape.Dimensions[2]);
            Assert.AreEqual(2, layer.Filters.Shape.Dimensions[3]);

            for (var i = 0; i < 8; i++)
            {
                layer.Filters.Set(0, 0, i, 0, i);
                layer.Filters.Set(0, 0, i, 1, i * 2.0);
            }

            var input = BuilderInstance.Volume.From(new[]
            {
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0,
                13.0, 14.0,
                15.0, 16.0
            }, new Shape(inputWidth, inputHeight, inputDepth, inputBatchSize));
            layer.DoForward(input);
        }

        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 15;
            const int inputHeight = 15;
            const int inputDepth = 2;

            const int batchSize = 3;

            var layer = new FullyConnLayer<double>(2) { BiasPref = 0.1 };

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth, batchSize);
        }

        [Test]
        public void GradientWrtParametersCheck()
        {
            const int inputWidth = 2;
            const int inputHeight = 2;
            const int inputDepth = 2;
            const int batchSize = 3;

            // Create layer
            var layer = new FullyConnLayer<double>(2) { BiasPref = 0.1 };

            GradientCheckTools.GradienWrtParameterstCheck(inputWidth, inputHeight, inputDepth, batchSize, layer);
        }
    }
}