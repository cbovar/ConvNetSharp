using System.Linq;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using NUnit.Framework;

namespace ConvNetSharp.Core.Tests
{
    [TestFixture]
    public class ConvLayerTests
    {
        [Test]
        public void ComputeTwiceGradientShouldYieldTheSameResult()
        {
            const int inputWidth = 10;
            const int inputHeight = 10;
            const int inputDepth = 2;

            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 2;

            // Create layer
            var layer = new ConvLayer<double>(filterWidth, filterHeight, filterCount) { Stride = 2, BiasPref = 0.1 };
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
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 15;
            const int inputHeight = 15;
            const int inputDepth = 2;

            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 5;

            const int batchSize = 3;

            // Create layer
            var layer = new ConvLayer<double>(filterWidth, filterHeight, filterCount) { Stride = 2, BiasPref = 0.1 };

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth, batchSize);
        }

        [Test]
        public void GradientWrtParametersCheck()
        {
            const int inputWidth = 10;
            const int inputHeight = 10;
            const int inputDepth = 2;

            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 2;

            const int batchSize = 1;

            // Create layer
            var layer = new ConvLayer<double>(filterWidth, filterHeight, filterCount) { Stride = 2 }; //BiasPref = 0.1 

            GradientCheckTools.GradienWrtParameterstCheck(inputWidth, inputHeight, inputDepth, batchSize, layer);
        }
    }
}