using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using NUnit.Framework;

namespace ConvNetSharp.Core.Tests
{
    [TestFixture]
    public class SoftmaxLayerTests
    {
        private readonly SoftmaxLayer layer;
        private readonly Volume<double> input;

        public VolumeBuilder<double> Volume => BuilderInstance.Volume;

        public SoftmaxLayerTests()
        {
            this.layer = new SoftmaxLayer(4);
            this.layer.Init(1, 1, 4);

            this.input = this.Volume.From(new[]
            {
                0.1, 0.1, 0.1, 0.1,
                1000, 2000, 3000, 4000,
                0, 0, 0, 0
            }, new Shape(1, 1, 4, 3));
        }

        [Test]
        public void OutputIsNormalized()
        {
            var output = this.layer.DoForward(this.input, true);
            Assert.AreEqual(1, output.Shape.Dimensions[0]);
            Assert.AreEqual(1, output.Shape.Dimensions[1]);
            Assert.AreEqual(4, output.Shape.Dimensions[2]);
            Assert.AreEqual(3, output.Shape.Dimensions[3]);

            var values = output.ToArray();
            Assert.AreEqual(0.25, values[0]);
            Assert.AreEqual(0.25, values[1]);
            Assert.AreEqual(0.25, values[2]);
            Assert.AreEqual(0.25, values[3]);

            Assert.AreEqual(0, values[4]);
            Assert.AreEqual(0, values[5]);
            Assert.AreEqual(0, values[6]);
            Assert.AreEqual(1, values[7]);

            Assert.AreEqual(0.25, values[8]);
            Assert.AreEqual(0.25, values[9]);
            Assert.AreEqual(0.25, values[10]);
            Assert.AreEqual(0.25, values[11]);
        }

        [Test]
        public void StorageIsReusedIfPossible()
        {
            var output1 = this.layer.DoForward(this.input, true);
            var output2 = this.layer.DoForward(this.input, true);
            Assert.AreSame(output1, output2, "Storage is reused if possible.");
        }
    }
}