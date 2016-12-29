using ConvNetSharp.Layers;
using NUnit.Framework;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class SoftmaxLayerTests
    {
        [Test]
        public void SerializationTest()
        {
            // Create a SoftmaxLayer
            var layer = new SoftmaxLayer(5);
            layer.Init(10, 10, 3);

            SoftmaxLayer desrialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                desrialized = formatter.Deserialize(ms) as SoftmaxLayer;
            }

            Assert.AreEqual(layer.InputDepth, desrialized.InputDepth);
            Assert.AreEqual(layer.InputHeight, desrialized.InputHeight);
            Assert.AreEqual(layer.InputWidth, desrialized.InputWidth);
            Assert.AreEqual(layer.OutputDepth, desrialized.OutputDepth);
            Assert.AreEqual(layer.OutputHeight, desrialized.OutputHeight);
            Assert.AreEqual(layer.OutputWidth, desrialized.OutputWidth);
            Assert.AreEqual(layer.ClassCount, desrialized.ClassCount);
        }
    }
}
