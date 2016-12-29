using ConvNetSharp.Layers;
using NUnit.Framework;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

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

        [Test]
        public void SerializationTest()
        {
            // Create a ReluLayer
            var layer = new ReluLayer();
            layer.Init(10, 10, 3);

            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                var result = formatter.Deserialize(ms) as ReluLayer;

                Assert.AreEqual(layer.InputDepth, result.InputDepth);
                Assert.AreEqual(layer.InputHeight, result.InputHeight);
                Assert.AreEqual(layer.InputWidth, result.InputWidth);
                Assert.AreEqual(layer.OutputDepth, result.OutputDepth);
                Assert.AreEqual(layer.OutputHeight, result.OutputHeight);
                Assert.AreEqual(layer.OutputWidth, result.OutputWidth);
            }
        }
    }
}