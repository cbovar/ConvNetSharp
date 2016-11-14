using ConvNetSharp.Layers;
using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class MaxoutLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            // Create layer
            const int groupSize = 4;
            var layer = new MaxoutLayer { GroupSize = groupSize };

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth);
        }
    }
}