using ConvNetSharp.Layers;
using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class TanhLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            // Create layer
            var layer = new TanhLayer();

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth);
        }
    }
}