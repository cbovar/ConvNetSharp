using System.Linq;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public class SingleOpTests : OpTests<float>
    {
        public SingleOpTests()
        {
            Op<float>.Count = 1;
            BuilderInstance<float>.Volume = new Volume.Single.VolumeBuilder();
        }

        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float)i).ToArray();
            return new Volume.Single.Volume(converted, shape);
        }
    }
}