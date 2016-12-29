using ConvNetSharp.Layers;
using ConvNetSharp.Serialization;
using Moq;
using NUnit.Framework;
using System;
using System.IO;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class NetTests
    {
        [Test]
        public void SerializerIsUsedTest()
        {
            var serializerMock = new Mock<INetSerializer>();

            var net = new Net();

            using (var ms = new MemoryStream())
            {
                // Serialization
                net.Save(serializerMock.Object, ms);
                serializerMock.Verify(o => o.Save(net, ms), Times.Once);

                // Deserialization
                ms.Position = 0;
                Net.Load(serializerMock.Object, ms);
                serializerMock.Verify(o => o.Load(ms), Times.Once);
            }
        }

        [Test]
        public void NonInputLayerAsFirstLayerShouldThrow()
        {
            var net = new Net();
            Assert.Throws(typeof(ArgumentException), () => net.AddLayer(new FullyConnLayer(10)));
        }

        [Test]
        public void AddingClassificationLayerWithoutPrecedingFullConnLayerShouldThrow()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(10, 10, 3));

            Assert.Throws(typeof(ArgumentException), () => net.AddLayer(new SoftmaxLayer(10)));

            Assert.Throws(typeof(ArgumentException), () => net.AddLayer(new SvmLayer(10)));
        }

        [Test]
        public void IncorrectNeuronCountWithClassificationLayer()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(10, 10, 3));
            net.AddLayer(new FullyConnLayer(5)); // should be 10

            Assert.Throws(typeof(ArgumentException), () => net.AddLayer(new SoftmaxLayer(10)));

            Assert.Throws(typeof(ArgumentException), () => net.AddLayer(new SvmLayer(10)));
        }

        [Test]
        public void AddingRegressionLayerWithoutPrecedingFullConnLayerShouldThrow()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(10, 10, 3));

            Assert.Throws(typeof(ArgumentException), () => net.AddLayer(new RegressionLayer(1)));
        }

        [Test]
        public void IncorrectNeuronCountWithRegressionLayer()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(10, 10, 3));
            net.AddLayer(new FullyConnLayer(5)); // should be 10

            Assert.Throws(typeof(ArgumentException), () => net.AddLayer(new RegressionLayer(10)));
        }

        [Test]
        public void BiasPrefUpdateWhenAddingReluLayer()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(10, 10, 3));
            var dotProduct1 = new FullyConnLayer(5);
            net.AddLayer(dotProduct1);
            net.AddLayer(new ReluLayer());
            var dotProduct2 = new ConvLayer(5,5,3);
            net.AddLayer(dotProduct2);
            net.AddLayer(new ReluLayer());

            Assert.AreEqual(0.1, dotProduct1.BiasPref);
            Assert.AreEqual(0.1, dotProduct2.BiasPref);
        }
    }
}
