using ConvNetSharp.Layers;
using ConvNetSharp.Serialization;
using NUnit.Framework;

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
    }
}
