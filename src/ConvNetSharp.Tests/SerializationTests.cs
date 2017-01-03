using ConvNetSharp.Fluent;
using ConvNetSharp.Layers;
using ConvNetSharp.Serialization;
using NUnit.Framework;
using System.IO;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class SerializationTests
    {
        [Test]
        public void JsonNetSerializerTest()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(5, 5, 3));
            net.AddLayer(new FullyConnLayer(3));
            net.AddLayer(new SoftmaxLayer(3));

            // Serialize to json
            var json = net.ToJSON();

            // Deserialize from json
            Net deserialized = SerializationExtensions.FromJSON(json);

            Assert.IsNotNull(deserialized.Layers);
            Assert.AreEqual(net.Layers.Count, deserialized.Layers.Count);
            Assert.IsTrue(net.Layers[0] is InputLayer);
            Assert.IsTrue(net.Layers[1] is FullyConnLayer);
            Assert.IsTrue(net.Layers[2] is SoftmaxLayer);
        }

        [Test]
        public void FluentJsonNetSerializerTest()
        {
            var net = FluentNet.Create(5, 5, 3)
                .FullyConn(3)
                .Softmax(3)
                .Build();

            // Serialize to json
            using (var ms = new MemoryStream())
            {
                net.SaveBinary(ms);
                ms.Position = 0;

                // Deserialize from json
                FluentNet deserialized = SerializationExtensions.LoadBinary(ms) as FluentNet;

                Assert.IsNotNull(deserialized);
                Assert.AreEqual(net.InputLayers.Count, deserialized.InputLayers.Count);
            }
        }

        [Test]
        public void BinaryNetSerializerTest()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(5, 5, 3));
            net.AddLayer(new FullyConnLayer(3));
            net.AddLayer(new SoftmaxLayer(3));

            // Serialize to json
            using (var ms = new MemoryStream())
            {
                net.SaveBinary(ms);
                ms.Position = 0;

                // Deserialize from json
                Net deserialized = SerializationExtensions.LoadBinary(ms) as Net;

                Assert.IsNotNull(deserialized.Layers);
                Assert.AreEqual(net.Layers.Count, deserialized.Layers.Count);
                Assert.IsTrue(net.Layers[0] is InputLayer);
                Assert.IsTrue(net.Layers[1] is FullyConnLayer);
                Assert.IsTrue(net.Layers[2] is SoftmaxLayer);
            }
        }
    }
}
