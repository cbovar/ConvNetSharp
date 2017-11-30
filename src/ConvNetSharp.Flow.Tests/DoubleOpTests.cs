using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
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