using ConvNetSharp.Layers;
using NUnit.Framework;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

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

        [Test]
        public void SerializationTest()
        {
            // Create a PoolLayer
            var layer = new PoolLayer(2,4)
            {
                Pad = 5, 
                Stride = 3
            };
            layer.Init(10, 10, 3);

            PoolLayer deserialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                deserialized = formatter.Deserialize(ms) as PoolLayer;
            }

            Assert.AreEqual(layer.InputDepth, deserialized.InputDepth);
            Assert.AreEqual(layer.InputHeight, deserialized.InputHeight);
            Assert.AreEqual(layer.InputWidth, deserialized.InputWidth);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.Height, deserialized.Height);
            Assert.AreEqual(layer.Width, deserialized.Width);
            Assert.AreEqual(layer.Pad, deserialized.Pad);
            Assert.AreEqual(layer.Stride, deserialized.Stride);
        }
    }
}