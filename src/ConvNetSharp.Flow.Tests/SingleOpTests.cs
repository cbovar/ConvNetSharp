using System.Linq;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Single;
using NUnit.Framework;

namespace ConvNetSharp.Flow.Tests
{
    [TestFixture]
    public class SingleOpTests : OpTests<float>
    {
        public SingleOpTests()
        {
            Op<float>.Count = 1;
            BuilderInstance<float>.Volume = new VolumeBuilder();
        }

        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float)i).ToArray();
            return BuilderInstance.Volume.From(converted, shape);
        }
    }
}