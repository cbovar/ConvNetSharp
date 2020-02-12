using System.Linq;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using NUnit.Framework;

namespace ConvNetSharp.Core.Tests
{
    [TestFixture]
    public class DropoutLayerTests
    {
        [Test]
        public void Learning()
        {
            var n = 1000000;
            var dropProbability = 0.2;
            var layer = new DropoutLayer<double>(dropProbability);
            layer.Init(1, 1, n);

            var input = BuilderInstance.Volume.From(new double[n].Populate(1.0), new Shape(1, 1, n, 1));
            var result = layer.DoForward(input, true);

            var val = result.ToArray().First(o => o != 0.0);
            var scalingFactor = 1.0 / (1.0 - dropProbability);
            Assert.AreEqual(scalingFactor, val); // Make sure output is scaled during learning

            var average = result.ToArray().Average();
            var measuredProba = average * dropProbability;
            Assert.AreEqual(dropProbability, measuredProba, 0.001); // Make sure dropout really happened
        }

        [Test]
        public void NotLearning()
        {
            var n = 1000000;
            var dropProbability = 0.2;
            var layer = new DropoutLayer<double>(dropProbability);
            layer.Init(1, 1, n);

            var input = BuilderInstance.Volume.From(new double[n].Populate(1.0), new Shape(1, 1, n, 1));
            var result = layer.DoForward(input);

            var average = result.ToArray().Average();
            Assert.AreEqual(1.0, average); // Let everything go through
        }
    }
}