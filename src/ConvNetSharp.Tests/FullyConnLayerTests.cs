using ConvNetSharp.Layers;
using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class FullyConnLayerTests
    {
        [Test]
        public void GradientWrtInputCheck2()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            // Create layer
            const int neuronCount = 20;
            var layer = new FullyConnLayer(neuronCount);

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth);
        }

        [Test]
        public void GradientWrtParametersCheck()
        {
            const int inputWidth = 2;
            const int inputHeight = 2;
            const int inputDepth = 2;
            const int neuronCount = 2;

            // Create layer
            var layer = new FullyConnLayer(neuronCount);
            layer.Init(inputWidth, inputHeight, inputDepth);

            GradientCheckTools.GradienWrtParameterstCheck(inputWidth, inputHeight, inputDepth, layer);
        }
    }
}