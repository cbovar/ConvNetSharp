using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class VolumeWrapperTests
    {
        [Test]
        public void CloneTest()
        {
            var vol1 = new Volume(10, 10, 10);
            var wrapper = new VolumeWrapper(new[] { vol1 });
            var clone = wrapper.Clone();

            for (int i = 0; i < wrapper.Length; i++)
            {
                Assert.AreEqual(wrapper.Get(i), clone.Get(i));
            }

            var vol2 = new Volume(20, 20, 20);
            var wrapper2 = new VolumeWrapper(new[] { vol1, vol2 });
            var clone2 = wrapper2.Clone();

            for (int i = 0; i < wrapper2.Length; i++)
            {
                Assert.AreEqual(wrapper2.Get(i), clone2.Get(i));
            }
        }
    }
}
