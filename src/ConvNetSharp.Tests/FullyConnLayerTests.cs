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

            FullyConnLayer desrialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                desrialized = formatter.Deserialize(ms) as FullyConnLayer;
            }

            Assert.AreEqual(layer.BiasPref, desrialized.BiasPref);
            Assert.AreEqual(layer.Filters.Count, desrialized.Filters.Count);
            Assert.AreEqual(layer.InputDepth, desrialized.InputDepth);
            Assert.AreEqual(layer.InputHeight, desrialized.InputHeight);
            Assert.AreEqual(layer.InputWidth, desrialized.InputWidth);
            Assert.AreEqual(layer.L1DecayMul, desrialized.L1DecayMul);
            Assert.AreEqual(layer.L2DecayMul, desrialized.L2DecayMul);
            Assert.AreEqual(layer.NeuronCount, desrialized.NeuronCount);
            Assert.AreEqual(layer.OutputDepth, desrialized.OutputDepth);
            Assert.AreEqual(layer.OutputHeight, desrialized.OutputHeight);
            Assert.AreEqual(layer.OutputWidth, desrialized.OutputWidth);

            for (int j = 0; j < layer.Filters.Count; j++)
            {
                var filter = layer.Filters[j];
                var deserializedFilter = desrialized.Filters[j];

                for (int i = 0; i < filter.Weights.Length; i++)
                {
                    Assert.AreEqual(filter.Weights[i], deserializedFilter.Weights[i]);
                }
            }

            for (int i = 0; i < layer.Biases.Weights.Length; i++)
            {
                Assert.AreEqual(layer.Biases.Weights[i], desrialized.Biases.Weights[i]);
            }
        }
    }
}