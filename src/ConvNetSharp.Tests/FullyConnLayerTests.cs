using ConvNetSharp.Layers;
using NUnit.Framework;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class FullyConnLayerTests
    {
        [Test]
        [TestCase(2, 2, 2, 2)]
        [TestCase(20, 20, 2, 20)]
        public void GradientWrtParametersCheck(int inputWidth, int inputHeight, int inputDepth, int neuronCount)
        {
            // Create layer
            var layer = new FullyConnLayer(neuronCount);

            GradientCheckTools.GradienWrtParameterstCheck(inputWidth, inputHeight, inputDepth, layer);
        }

        [Test]
        public void SerializationTest()
        {
            // Create a FullyConnLayer
            var layer = new FullyConnLayer(20)
            {
                Activation = Activation.Relu,
                BiasPref = 0.1,
            };
            layer.Init(10, 10, 3);

            foreach (var filter in layer.Filters)
            {
                for (int i = 0; i < filter.Weights.Length; i++)
                {
                    filter.Weights[i] = i;
                }
            }

            for (int i = 0; i < layer.Biases.Weights.Length; i++)
            {
                layer.Biases.Weights[i] = i;
            }

            FullyConnLayer deserialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                deserialized = formatter.Deserialize(ms) as FullyConnLayer;
            }

            Assert.AreEqual(layer.Activation, deserialized.Activation);
            Assert.AreEqual(layer.BiasPref, deserialized.BiasPref);
            Assert.AreEqual(layer.Filters.Count, deserialized.Filters.Count);
            Assert.AreEqual(layer.InputDepth, deserialized.InputDepth);
            Assert.AreEqual(layer.InputHeight, deserialized.InputHeight);
            Assert.AreEqual(layer.InputWidth, deserialized.InputWidth);
            Assert.AreEqual(layer.L1DecayMul, deserialized.L1DecayMul);
            Assert.AreEqual(layer.L2DecayMul, deserialized.L2DecayMul);
            Assert.AreEqual(layer.NeuronCount, deserialized.NeuronCount);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);

            for (int j = 0; j < layer.Filters.Count; j++)
            {
                var filter = layer.Filters[j];
                var deserializedFilter = deserialized.Filters[j];

                for (int i = 0; i < filter.Weights.Length; i++)
                {
                    Assert.AreEqual(filter.Weights[i], deserializedFilter.Weights[i]);
                }
            }

            for (int i = 0; i < layer.Biases.Weights.Length; i++)
            {
                Assert.AreEqual(layer.Biases.Weights[i], deserialized.Biases.Weights[i]);
            }
        }
    }
}