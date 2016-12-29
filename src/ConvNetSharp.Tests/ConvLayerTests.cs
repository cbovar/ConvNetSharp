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

            ConvLayer desrialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                desrialized = formatter.Deserialize(ms) as ConvLayer;
            }

            Assert.AreEqual(layer.Activation, desrialized.Activation);
            Assert.AreEqual(layer.BiasPref, desrialized.BiasPref);
            Assert.AreEqual(layer.Stride, desrialized.Stride);
            Assert.AreEqual(layer.Pad, desrialized.Pad);
            Assert.AreEqual(layer.Filters.Count, desrialized.Filters.Count);
            Assert.AreEqual(layer.InputDepth, desrialized.InputDepth);
            Assert.AreEqual(layer.InputHeight, desrialized.InputHeight);
            Assert.AreEqual(layer.InputWidth, desrialized.InputWidth);
            Assert.AreEqual(layer.L1DecayMul, desrialized.L1DecayMul);
            Assert.AreEqual(layer.L2DecayMul, desrialized.L2DecayMul);
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