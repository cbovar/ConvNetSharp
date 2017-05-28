using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.Tests
{
    [TestClass]
    public class DoubleVolumeTests : VolumeTests<double>
    {
        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new Double.Volume(values, shape);
        }
    }
}