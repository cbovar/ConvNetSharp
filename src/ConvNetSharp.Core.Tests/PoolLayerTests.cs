using System.Linq;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using NUnit.Framework;

namespace ConvNetSharp.Core.Tests
{
    [TestFixture]
    public class PoolLayerTests
    {
        [Test]
        public void ComputeTwiceGradientShouldYieldTheSameResult()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            const int width = 2;
            const int height = 2;

            // Create layer
            var layer = new PoolLayer<double>(width, height) { Stride = 2 };
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
            const int inputWidth = 4;
            const int inputHeight = 4;
            const int inputDepth = 4;
            const int inputBatchSize = 4;

            var layer = new PoolLayer<double>(2, 2);
            layer.Init(inputWidth, inputHeight, inputDepth);

            var data = new double[4 * 4 * 4 * 4];
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = i;
            }

            var input = BuilderInstance.Volume.From(data, new Shape(inputWidth, inputHeight, inputDepth, inputBatchSize));
            layer.DoForward(input);

            Assert.AreEqual(2, layer.OutputActivation.Shape.Dimensions[0]);
            Assert.AreEqual(2, layer.OutputActivation.Shape.Dimensions[1]);
            Assert.AreEqual(4, layer.OutputActivation.Shape.Dimensions[2]);
            Assert.AreEqual(4, layer.OutputActivation.Shape.Dimensions[3]);

            Assert.AreEqual(5.0, layer.OutputActivation.Get(0, 0, 0, 0));
            Assert.AreEqual(21.0, layer.OutputActivation.Get(0, 0, 1, 0));
            Assert.AreEqual(85.0, layer.OutputActivation.Get(0, 0, 1, 1));
        }

        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            const int width = 2;
            const int height = 2;

            const int batchSize = 3;

            // Create layer
            var layer = new PoolLayer<double>(width, height) { Stride = 2 };

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth, batchSize, 1e-6);
        }
    }
}