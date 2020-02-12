using ConvNetSharp.Volume.GPU.Double;
using ConvNetSharp.Volume.Tests;
using NUnit.Framework;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestFixture]
    public class DoubleVolumeTests : VolumeTests<double>
    {
        public DoubleVolumeTests()
        {
            BuilderInstance<double>.Volume = new VolumeBuilder();
        }

        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new Double.Volume(values, shape);
        }
    }
}