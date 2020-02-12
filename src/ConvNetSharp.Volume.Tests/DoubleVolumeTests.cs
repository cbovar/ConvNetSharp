using ConvNetSharp.Volume.Double;
using NUnit.Framework;

namespace ConvNetSharp.Volume.Tests
{
    [TestFixture]
    public class DoubleVolumeTests : VolumeTests<double>
    {
        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return BuilderInstance.Volume.From(values, shape);
        }
    }
}