using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestClass]
    public class GenericVolumeTests
    {
        [TestMethod]
        public void BuildVolumeFromStorageAndShape()
        {
            var shape = new Shape(2, 2);
            var storage = new Double.VolumeStorage(new[] { 1.0, 2.0, 3.0, 4.0 }, shape, GpuContext.Default);
            var volume = BuilderInstance<double>.Volume.Build(storage, shape);

            Assert.IsTrue(storage.ToArray().SequenceEqual(volume.Storage.ToArray()));
        }

        [TestMethod]
        public void ClearOnDevice()
        {
            int l = 4080;
            var shape = new Shape(l);
            var data = new double[l].Populate(1.0);
            var storage = new Double.VolumeStorage(data, shape, GpuContext.Default);

            // Copy to device
            storage.CopyToDevice();

            //Clear
            storage.Clear();

            // Copy back to host
            storage.CopyToHost();
            Assert.IsTrue(storage.ToArray().All(o => o == 0.0));
        }

        [TestMethod]
        public void ClearOnHost()
        {
            int l = 4080;
            var shape = new Shape(l);
            var data = new double[l].Populate(1.0);
            var storage = new Double.VolumeStorage(data, shape, GpuContext.Default);

            //Clear
            storage.Clear();

            // Copy back to host
            storage.CopyToHost();
            Assert.IsTrue(storage.ToArray().All(o => o == 0.0));
        }

        [TestMethod]
        public void CopyToHostAndDevice()
        {
            int l = 4080;
            var shape = new Shape(l);
            var data = new double[l].Populate(1.0);
            var storage = new Double.VolumeStorage(data, shape, GpuContext.Default);

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

        [ClassInitialize]
        public static void ClassInit(TestContext context)
        {
            BuilderInstance<double>.Volume = new Double.VolumeBuilder();
        }

        [TestMethod]
        public void ReShape_UnknownDimension()
        {
            var volume = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3), GpuContext.Default);

            var reshaped = volume.ReShape(1, -1);
            Assert.AreEqual(reshaped.Shape.DimensionCount, 2);
            Assert.AreEqual(reshaped.Shape.TotalLength, 3);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException), "Imcompatible dimensions provided")]
        public void ReShape_WrongDimension()
        {
            var volume = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3), GpuContext.Default);
            volume.ReShape(1, 4);
        }

        [TestMethod]
        public void ReShape_Data()
        {
            var data = new[] { 1.0, 2.0, 3.0 };
            var volume = new Double.Volume(data, new Shape(3), GpuContext.Default);

            var reshaped = volume.ReShape(1, -1);

            Assert.IsTrue(reshaped.ToArray().SequenceEqual(volume.Storage.ToArray()));
        }
    }
}