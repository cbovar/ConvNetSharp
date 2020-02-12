using System.Linq;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using NUnit.Framework;

namespace ConvNetSharp.Core.Tests
{
    [TestFixture]
    public class ReluLayerTests
    {
        [Test]
        public void ComputeTwiceGradientShouldYieldTheSameResult()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            var layer = new ReluLayer();
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = BuilderInstance.Volume.Random(new Shape(inputWidth, inputHeight, inputDepth));
            var output = layer.DoForward(input, true);

            // Set output gradients to 1
            var outputGradient = BuilderInstance.Volume.From(new double[output.Shape.TotalLength].Populate(1.0), output.Shape);

            // Backward pass to retrieve gradients
            layer.Backward(outputGradient);
            var step1 = ((Volume.Double.Volume)layer.InputActivationGradients.Clone()).ToArray();

            layer.Backward(outputGradient);
            var step2 = ((Volume.Double.Volume)layer.InputActivationGradients.Clone()).ToArray();

            Assert.IsTrue(step1.SequenceEqual(step2));
        }

        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            const int batchSize = 3;

            // Create layer
            var layer = new ReluLayer();

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth, batchSize, 1e-6);
        }
    }
}