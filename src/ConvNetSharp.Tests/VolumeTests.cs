using NUnit.Framework;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class VolumeTests
    {
        [Test]
        public void FlipTest()
        {
            var volume = new Volume(3, 3, 2, 0.0);
            volume.Set(1, 1, 0, 1.0);
            volume.Set(1, 1, 1, 2.0);

            volume.Set(0, 0, 0, 10.0);
            volume.Set(0, 0, 1, 15.0);

            volume.Set(0, 2, 0, 20.0);
            volume.Set(0, 2, 1, 25.0);

            volume.Set(2, 0, 0, 30.0);
            volume.Set(2, 0, 1, 35.0);

            volume.Set(2, 2, 0, 40.0);
            volume.Set(2, 2, 1, 45.0);

            var flipped = VolumeUtilities.Flip(volume, VolumeUtilities.FlipMode.LeftRight);
            Assert.AreEqual(1.0, flipped.Get(1, 1, 0));
            Assert.AreEqual(2.0, flipped.Get(1, 1, 1));

            Assert.AreEqual(10.0, flipped.Get(2, 0, 0));
            Assert.AreEqual(15.0, flipped.Get(2, 0, 1));

            Assert.AreEqual(20.0, flipped.Get(2, 2, 0));
            Assert.AreEqual(25.0, flipped.Get(2, 2, 1));

            Assert.AreEqual(30.0, flipped.Get(0, 0, 0));
            Assert.AreEqual(35.0, flipped.Get(0, 0, 1));

            Assert.AreEqual(40.0, flipped.Get(0, 2, 0));
            Assert.AreEqual(45.0, flipped.Get(0, 2, 1));

            flipped = VolumeUtilities.Flip(volume, VolumeUtilities.FlipMode.UpDown);
            Assert.AreEqual(1.0, flipped.Get(1, 1, 0));
            Assert.AreEqual(2.0, flipped.Get(1, 1, 1));

            Assert.AreEqual(10.0, flipped.Get(0, 2, 0));
            Assert.AreEqual(15.0, flipped.Get(0, 2, 1));

            Assert.AreEqual(20.0, flipped.Get(0, 0, 0));
            Assert.AreEqual(25.0, flipped.Get(0, 0, 1));

            Assert.AreEqual(30.0, flipped.Get(2, 2, 0));
            Assert.AreEqual(35.0, flipped.Get(2, 2, 1));

            Assert.AreEqual(40.0, flipped.Get(2, 0, 0));
            Assert.AreEqual(45.0, flipped.Get(2, 0, 1));

            flipped = VolumeUtilities.Flip(volume, VolumeUtilities.FlipMode.Both);
            Assert.AreEqual(1.0, flipped.Get(1, 1, 0));
            Assert.AreEqual(2.0, flipped.Get(1, 1, 1));

            Assert.AreEqual(10.0, flipped.Get(2, 2, 0));
            Assert.AreEqual(15.0, flipped.Get(2, 2, 1));

            Assert.AreEqual(20.0, flipped.Get(2, 0, 0));
            Assert.AreEqual(25.0, flipped.Get(2, 0, 1));

            Assert.AreEqual(30.0, flipped.Get(0, 2, 0));
            Assert.AreEqual(35.0, flipped.Get(0, 2, 1));

            Assert.AreEqual(40.0, flipped.Get(0, 0, 0));
            Assert.AreEqual(45.0, flipped.Get(0, 0, 1));
        }

        [Test]
        public void SerializationTest()
        {
            var volume = new Volume(30, 30, 30); // filled with random values
            for (int i = 0; i < volume.WeightGradients.Length; i++)
            {
                volume.WeightGradients[i] = i;
            }

            Volume deserialized;
            using (var ms = new MemoryStream())
            {
                // Serialize
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(ms, volume);

                // Deserialize
                ms.Position = 0;
                deserialized = formatter.Deserialize(ms) as Volume;
            }

            Assert.AreEqual(volume.Weights.Length, deserialized.Weights.Length);

            for (int i = 0; i < volume.Weights.Length; i++)
            {
                Assert.AreEqual(volume.Weights[i], deserialized.Weights[i]);
                Assert.AreEqual(volume.WeightGradients[i], deserialized.WeightGradients[i]);
            }
        }
    }
}
