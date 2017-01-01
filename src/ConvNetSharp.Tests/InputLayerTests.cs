using ConvNetSharp.Layers;
using NUnit.Framework;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class InputLayerTests
    {
        [Test]
        public void SerializationTest()
        {
            // Create a InputLayer
            var layer = new InputLayer(5, 5, 3);
            layer.Init(10, 10, 3);

            InputLayer deserialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                deserialized = formatter.Deserialize(ms) as InputLayer;
            }

            Assert.AreEqual(layer.InputDepth, deserialized.InputDepth);
            Assert.AreEqual(layer.InputHeight, deserialized.InputHeight);
            Assert.AreEqual(layer.InputWidth, deserialized.InputWidth);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
        }
    }
}
