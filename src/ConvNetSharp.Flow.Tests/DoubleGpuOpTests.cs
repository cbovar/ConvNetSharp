using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public class DoubleGpuOpTests : OpTests<double>
    {
        public DoubleGpuOpTests()
        {
            Op<double>.Count = 1;
            BuilderInstance<double>.Volume = new Volume.GPU.Double.VolumeBuilder();
        }

        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new Volume.GPU.Double.Volume(values, shape);
        }
    }
}