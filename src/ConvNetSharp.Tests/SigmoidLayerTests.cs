using ConvNetSharp.Layers;
using NUnit.Framework;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class SigmoidLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            // Create layer
            var layer = new SigmoidLayer();

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth);
        }

        [Test]
        public void SerializationTest()
        {
            // Create a SigmoidLayer
            var layer = new SigmoidLayer();
            layer.Init(10, 10, 3);

            SigmoidLayer desrialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);

                // Deserialize
                ms.Position = 0;
                desrialized = formatter.Deserialize(ms) as SigmoidLayer;
            }

            Assert.AreEqual(layer.InputDepth, desrialized.InputDepth);
            Assert.AreEqual(layer.InputHeight, desrialized.InputHeight);
            Assert.AreEqual(layer.InputWidth, desrialized.InputWidth);
            Assert.AreEqual(layer.OutputDepth, desrialized.OutputDepth);
            Assert.AreEqual(layer.OutputHeight, desrialized.OutputHeight);
            Assert.AreEqual(layer.OutputWidth, desrialized.OutputWidth);
         }
    }
}