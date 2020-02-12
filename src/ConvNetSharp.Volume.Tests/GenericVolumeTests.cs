using System;
using System.Linq;
using ConvNetSharp.Volume.Double;
using NUnit.Framework;

namespace ConvNetSharp.Volume.Tests
{
    [TestFixture]
    public class GenericVolumeTests
    {
        [Test]
        public void BuildVolumeFromStorageAndShape()
        {
            var shape = new Shape(2, 2);
            var storage = new NcwhVolumeStorage<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, shape);
            var volume = BuilderInstance<double>.Volume.Build(storage, shape);

            Assert.IsTrue(storage.ToArray().SequenceEqual(volume.Storage.ToArray()));
        }

        [Test]
        public void ReShape_UnknownDimension()
        {
            var volume = BuilderInstance.Volume.From(new[] { 1.0, 2.0, 3.0 }, new Shape(3));

            var reshaped = volume.ReShape(1, -1);
            Assert.AreEqual(reshaped.Shape.TotalLength, 3);
        }

        [Test]
        public void ReShape_WrongDimension()
        {
            var volume = BuilderInstance.Volume.From(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            Assert.Throws<ArgumentException>(() => volume.ReShape(1, 4));
        }

        [Test]
        public void ReShapeKeep()
        {
            var volume = BuilderInstance.Volume.From(new[]
            {
                1.0, 2.0,
                3.0, 4.0,

                5.0, 6.0,
                7.0, 8.0,

                9.0, 10.0,
                11.0, 12.0
            }, new Shape(2, 2, 1, 3));

            var reshaped = volume.ReShape(1, 1, Shape.None, Shape.Keep);

            Assert.AreEqual(reshaped.Shape.TotalLength, 12);
            Assert.AreEqual(new Shape(1, 1, 4, 3), reshaped.Shape);
        }
    }
}