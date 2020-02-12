using System;
using System.Linq;
using ConvNetSharp.Volume.GPU.Double;
using NUnit.Framework;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestFixture]
    public class GenericVolumeTests
    {
        [OneTimeSetUp]
        public void ClassInit()
        {
            BuilderInstance<double>.Volume = new VolumeBuilder();
        }

        [Test]
        public void BuildVolumeFromStorageAndShape()
        {
            var shape = new Shape(2, 2);
            var storage = new VolumeStorage(new[] { 1.0, 2.0, 3.0, 4.0 }, shape, GpuContext.Default);
            var volume = BuilderInstance<double>.Volume.Build(storage, shape);

            Assert.IsTrue(storage.ToArray().SequenceEqual(volume.Storage.ToArray()));
        }

        [Test]
        public void ClearOnDevice()
        {
            const int l = 4080;
            var shape = new Shape(l);
            var data = new double[l].Populate(1.0);
            var storage = new VolumeStorage(data, shape, GpuContext.Default);

            // Copy to device
            storage.CopyToDevice();

            //Clear
            storage.Clear();

            // Copy back to host
            storage.CopyToHost();
            Assert.IsTrue(storage.ToArray().All(o => o == 0.0));
        }

        [Test]
        public void ClearOnHost()
        {
            const int l = 4080;
            var shape = new Shape(l);
            var data = new double[l].Populate(1.0);
            var storage = new VolumeStorage(data, shape, GpuContext.Default);

            //Clear
            storage.Clear();

            // Copy back to host
            storage.CopyToHost();
            Assert.IsTrue(storage.ToArray().All(o => o == 0.0));
        }

        [Test]
        public void CopyToHostAndDevice()
        {
            const int l = 4080;
            var shape = new Shape(l);
            var data = new double[l].Populate(1.0);
            var storage = new VolumeStorage(data, shape, GpuContext.Default);

            Assert.IsTrue(data.SequenceEqual(storage.ToArray()));
            Assert.AreEqual(DataLocation.Host, storage.Location);

            // Copy to device
            storage.CopyToDevice();
            Assert.AreEqual(DataLocation.Device, storage.Location);

            // Copy back to host
            storage.CopyToHost();
            Assert.IsTrue(data.SequenceEqual(storage.ToArray()));
            Assert.AreEqual(DataLocation.Host, storage.Location);
        }

        [Test]
        public void ReShape_Data()
        {
            var data = new[] { 1.0, 2.0, 3.0 };
            var volume = new Double.Volume(data, new Shape(3), GpuContext.Default);

            var reshaped = volume.ReShape(1, -1);

            Assert.IsTrue(reshaped.ToArray().SequenceEqual(volume.Storage.ToArray()));
        }

        [Test]
        public void ReShape_UnknownDimension()
        {
            var volume = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3), GpuContext.Default);

            var reshaped = volume.ReShape(1, -1);
            Assert.AreEqual(reshaped.Shape.TotalLength, 3);
        }

        [Test]
        public void ReShape_WrongDimension()
        {
            var volume = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3), GpuContext.Default);
            Assert.Throws<ArgumentException>(() => volume.ReShape(1, 4));
        }
    }
}