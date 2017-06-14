using ConvNetSharp.Volume.GPU.Double;
using ConvNetSharp.Volume.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestClass]
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