using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class ConvLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 30;
            const int inputHeight = 30;
            const int inputDepth = 2;

            // Create layer
            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 2;

            var layer = new ConvLayer(filterWidth, filterHeight, filterCount);

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth);
        }

        [Test]
        public void GradientWrtParametersCheck()
        {
            const int inputWidth = 30;
            const int inputHeight = 30;
            const int inputDepth = 2;

            // Create layer
            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 2;

            var layer = new ConvLayer(filterWidth, filterHeight, filterCount);

            GradientCheckTools.GradienWrtParameterstCheck(inputWidth, inputHeight, inputDepth, layer);
        }
    }
}