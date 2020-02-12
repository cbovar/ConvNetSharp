using ConvNetSharp.Core.Layers.Double;
using NUnit.Framework;

namespace ConvNetSharp.Core.Tests
{
    [TestFixture]
    public class RegressionLayerTests
    {
        [Test]
        public void Instantiation()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            var layer = new ReluLayer();
            layer.Init(inputWidth, inputHeight, inputDepth);

            Assert.AreEqual(20, layer.OutputWidth);
            Assert.AreEqual(20, layer.OutputHeight);
            Assert.AreEqual(2, layer.OutputDepth);
        }
    }
}