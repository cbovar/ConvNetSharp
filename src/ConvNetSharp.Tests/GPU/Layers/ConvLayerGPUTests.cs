using ConvNetSharp.GPU.Layers;
using NUnit.Framework;
using System;
using System.IO;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class ConvLayerGPUTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            Environment.CurrentDirectory = Path.GetDirectoryName(typeof(SerializationTests).Assembly.Location);

            const int inputWidth = 30;
            const int inputHeight = 30;
            const int inputDepth = 2;

            // Create layer
            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 5;

            var layer = new ConvLayerGPU(filterWidth, filterHeight, filterCount) { Stride = 2 };

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth);
        }

        [Test]
        public void GradientWrtParametersCheck()
        {
            Environment.CurrentDirectory = Path.GetDirectoryName(typeof(SerializationTests).Assembly.Location);

            const int inputWidth = 10;
            const int inputHeight = 10;
            const int inputDepth = 2;

            // Create layer
            const int filterWidth = 3;
            const int filterHeight = 3;
            const int filterCount = 2;

            var layer = new ConvLayerGPU(filterWidth, filterHeight, filterCount) { Stride = 2 };

            GradientCheckTools.GradienWrtParameterstCheck(inputWidth, inputHeight, inputDepth, layer);
        }

    }
}