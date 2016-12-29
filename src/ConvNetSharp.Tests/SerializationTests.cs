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
        public void BinaryNetSerializerTest()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(5, 5, 3));
            net.AddLayer(new FullyConnLayer(3));
            net.AddLayer(new SoftmaxLayer(3));

            var serializer = new BinaryNetSerializer();

            Net deserialized;
            using (var ms = new MemoryStream())
            {
                net.Save(serializer, ms);

                ms.Position = 0;
                deserialized = Net.Load(serializer, ms);
            }

            Assert.IsNotNull(deserialized.Layers);
            Assert.AreEqual(net.Layers.Count, deserialized.Layers.Count);
        }
    }
}
