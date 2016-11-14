using ConvNetSharp.Layers;
using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class PoolLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            // Create layer
            const int width = 2;
            const int height = 2;
            var layer = new PoolLayer(width, height) { Stride = 2 };

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth, 1e-6);
        }
    }
}