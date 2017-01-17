using ConvNetSharp.Layers;
using NUnit.Framework;
using System;
using System.IO;

namespace ConvNetSharp.GPU.Tests
{
    [TestFixture]
    public class GPUExtensionsTests
    {
        [Test]
        public void ToGpuNet()
        {
            Environment.CurrentDirectory = Path.GetDirectoryName(typeof(GPUExtensionsTests).Assembly.Location);

            var net = new Net();
            net.AddLayer(new InputLayer(5, 5, 3));
            var conv = new ConvLayer(2, 2, 16);
            net.AddLayer(conv);
            var fullycon = new FullyConnLayer(3);
            net.AddLayer(fullycon);
            net.AddLayer(new SoftmaxLayer(3));

            var gpuNet = net.ToGPU();
            var nonGpuNet = net.ToNonGPU();

            //TODO: actually test something
        }
    }
}
