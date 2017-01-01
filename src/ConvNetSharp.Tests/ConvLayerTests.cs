using ConvNetSharp.Layers;
using NUnit.Framework;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class ConvLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 30;
            const int inputHeight = 30;
            const int inputDepth = 2;

            // Create layer
            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 5;

            var layer = new ConvLayer(filterWidth, filterHeight, filterCount) { Stride = 2 };

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth);
        }

        [Test]
        public void GradientWrtParametersCheck()
        {
            const int inputWidth = 10;
            const int inputHeight = 10;
            const int inputDepth = 2;

            // Create layer
            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 2;

            var layer = new ConvLayer(filterWidth, filterHeight, filterCount) { Stride = 2 };

            GradientCheckTools.GradienWrtParameterstCheck(inputWidth, inputHeight, inputDepth, layer);
        }

        [Test]
        public void SerializationTest()
        {
            // Create a ConvLayer
            var layer = new ConvLayer(5,5,2)
            {
                Activation = Activation.Relu,
                BiasPref = 0.1,
                Pad = 1,
                Stride = 2
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

            ConvLayer deserialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                deserialized = formatter.Deserialize(ms) as ConvLayer;
            }

            Assert.AreEqual(layer.Activation, deserialized.Activation);
            Assert.AreEqual(layer.BiasPref, deserialized.BiasPref);
            Assert.AreEqual(layer.Stride, deserialized.Stride);
            Assert.AreEqual(layer.Pad, deserialized.Pad);
            Assert.AreEqual(layer.Filters.Count, deserialized.Filters.Count);
            Assert.AreEqual(layer.InputDepth, deserialized.InputDepth);
            Assert.AreEqual(layer.InputHeight, deserialized.InputHeight);
            Assert.AreEqual(layer.InputWidth, deserialized.InputWidth);
            Assert.AreEqual(layer.L1DecayMul, deserialized.L1DecayMul);
            Assert.AreEqual(layer.L2DecayMul, deserialized.L2DecayMul);
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