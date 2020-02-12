using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using NUnit.Framework;

namespace ConvNetSharp.Flow.Tests
{
    [TestFixture]
    public class DoubleOpTests : OpTests<double>
    {
        public DoubleOpTests()
        {
            Op<double>.Count = 1;
            BuilderInstance<double>.Volume = new VolumeBuilder();
        }

        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return BuilderInstance.Volume.From(values, shape);
        }
    }
}