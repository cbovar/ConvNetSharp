using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.GPU.Double;
using NUnit.Framework;

namespace ConvNetSharp.Flow.Tests
{
    [TestFixture]
    public class DoubleGpuOpTests : OpTests<double>
    {
        public DoubleGpuOpTests()
        {
            Op<double>.Count = 1;
            BuilderInstance<double>.Volume = new VolumeBuilder();
        }

        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new Volume.GPU.Double.Volume(values, shape);
        }
    }
}