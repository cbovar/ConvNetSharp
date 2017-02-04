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
            var conv = new ConvLayer(2, 2, 16);
            net.AddLayer(conv);
            var pool = new PoolLayer(2, 2);
            net.AddLayer(pool);
            var fullycon = new FullyConnLayer(3);
            net.AddLayer(fullycon);
            net.AddLayer(new SoftmaxLayer(3));

            // Serialize to json
            var json = net.ToJSON();

            // Deserialize from json
            Net deserialized = SerializationExtensions.FromJSON(json);

            // Make sure deserialized is identical to serialized
            Assert.IsNotNull(deserialized.Layers);
            Assert.AreEqual(net.Layers.Count, deserialized.Layers.Count);
            Assert.IsTrue(net.Layers[0] is InputLayer);

            var deserializedConv = net.Layers[1] as ConvLayer;
            Assert.NotNull(deserializedConv);
            Assert.NotNull(deserializedConv.Filters);
            Assert.AreEqual(16, deserializedConv.Filters.Count);
            for (int i = 0; i < deserializedConv.Filters.Count; i++)
            {
                for (int k = 0; k < deserializedConv.Filters[i].Length; k++)
                {
                    Assert.AreEqual(conv.Filters[i].Get(k), deserializedConv.Filters[i].Get(k));
                    Assert.AreEqual(conv.Filters[i].GetGradient(k), deserializedConv.Filters[i].GetGradient(k));
                }
            }

            var deserializedPool = net.Layers[2] as PoolLayer;
            Assert.NotNull(deserializedPool);
            Assert.AreEqual(2, deserializedPool.Height);
            Assert.AreEqual(2, deserializedPool.Width);
            Assert.AreEqual(2, deserializedPool.OutputHeight);
            Assert.AreEqual(2, deserializedPool.OutputWidth);
            Assert.AreEqual(0, deserializedPool.Pad);
            Assert.AreEqual(2, deserializedPool.Stride);

            var deserializedFullyCon = net.Layers[3] as FullyConnLayer;
            Assert.NotNull(deserializedFullyCon);
            Assert.NotNull(deserializedFullyCon.Filters);
            Assert.AreEqual(3, deserializedFullyCon.Filters.Count);
            for (int i = 0; i < deserializedFullyCon.Filters.Count; i++)
            {
                for (int k = 0; k < deserializedFullyCon.Filters[i].Length; k++)
                {
                    Assert.AreEqual(fullycon.Filters[i].Get(k), deserializedFullyCon.Filters[i].Get(k));
                    Assert.AreEqual(fullycon.Filters[i].GetGradient(k), deserializedFullyCon.Filters[i].GetGradient(k));
                }
            }

            Assert.IsTrue(net.Layers[4] is SoftmaxLayer);
            Assert.AreEqual(3, ((SoftmaxLayer)net.Layers[4]).ClassCount);
        }

        [Test]
        public void FluentBinaryNetSerializerTest()
        {
            var net = FluentNet.Create(5, 5, 3)
                .Conv(2, 2, 16)
                .FullyConn(3)
                .Softmax(3)
                .Build();

            // Serialize (binary)
            using (var ms = new MemoryStream())
            {
                net.SaveBinary(ms);
                ms.Position = 0;

                // Deserialize (binary)
                FluentNet deserialized = SerializationExtensions.LoadBinary(ms) as FluentNet;

                Assert.IsNotNull(deserialized);
                Assert.AreEqual(net.InputLayers.Count, deserialized.InputLayers.Count);

                // TODO: improve test
            }
        }

        [Test]
        public void BinaryNetSerializerTest()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(5, 5, 3));
            var conv = new ConvLayer(2, 2, 16);
            net.AddLayer(conv);
            var fullycon = new FullyConnLayer(3);
            net.AddLayer(fullycon);
            net.AddLayer(new SoftmaxLayer(3));

            // Serialize (binary)
            using (var ms = new MemoryStream())
            {
                net.SaveBinary(ms);
                ms.Position = 0;

                // Deserialize (binary)
                Net deserialized = SerializationExtensions.LoadBinary(ms) as Net;

                // Make sure deserialized is identical to serialized
                Assert.IsNotNull(deserialized.Layers);
                Assert.AreEqual(net.Layers.Count, deserialized.Layers.Count);
                Assert.IsTrue(net.Layers[0] is InputLayer);

                var deserializedConv = net.Layers[1] as ConvLayer;
                Assert.NotNull(deserializedConv);
                Assert.NotNull(deserializedConv.Filters);
                Assert.AreEqual(16, deserializedConv.Filters.Count);
                for (int i = 0; i < deserializedConv.Filters.Count; i++)
                {
                    for (int k = 0; k < deserializedConv.Filters[i].Length; k++)
                    {
                        Assert.AreEqual(conv.Filters[i].Get(k), deserializedConv.Filters[i].Get(k));
                        Assert.AreEqual(conv.Filters[i].GetGradient(k), deserializedConv.Filters[i].GetGradient(k));
                    }
                }

                var deserializedFullyCon = net.Layers[2] as FullyConnLayer;
                Assert.NotNull(deserializedFullyCon);
                Assert.NotNull(deserializedFullyCon.Filters);
                Assert.AreEqual(3, deserializedFullyCon.Filters.Count);
                for (int i = 0; i < deserializedFullyCon.Filters.Count; i++)
                {
                    for (int k = 0; k < deserializedFullyCon.Filters[i].Length; k++)
                    {
                        Assert.AreEqual(fullycon.Filters[i].Get(k), deserializedFullyCon.Filters[i].Get(k));
                        Assert.AreEqual(fullycon.Filters[i].GetGradient(k), deserializedFullyCon.Filters[i].GetGradient(k));
                    }
                }

                Assert.IsTrue(net.Layers[3] is SoftmaxLayer);
                Assert.AreEqual(3, ((SoftmaxLayer)net.Layers[3]).ClassCount);
            }
        }
    }
}
