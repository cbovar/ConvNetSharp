using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class ReluLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            // Create layer
            var layer = new ReluLayer();

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth, 1e-6);
        }
    }
}