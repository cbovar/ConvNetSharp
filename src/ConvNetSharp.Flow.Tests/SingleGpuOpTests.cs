using System.Linq;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public class SingleGpuOpTests : OpTests<float>
    {
        public SingleGpuOpTests()
        {
            Op<float>.Count = 1;
            BuilderInstance<float>.Volume = new Volume.GPU.Single.VolumeBuilder();
        }

        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float)i).ToArray();
            return new Volume.GPU.Single.Volume(converted, shape);
        }
    }
}