using System;
using System.Linq;
using ConvNetSharp.Volume.GPU.Double;
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
            var storage = new VolumeStorage(new[] { 1.0, 2.0, 3.0, 4.0 }, shape, GpuContext.Default);
            var volume = BuilderInstance<double>.Volume.Build(storage, shape);

            Assert.IsTrue(storage.ToArray().SequenceEqual(volume.Storage.ToArray()));
        }

        [ClassInitialize]
        public static void ClassInit(TestContext context)
        {
            BuilderInstance<double>.Volume = new VolumeBuilder();
        }

        [TestMethod]
        public void CopyToHostAndDevice()
        {
            var l = 4080;
            var shape = new Shape(l);
            var data = new double[l].Populate(1.0);
            var storage = new VolumeStorage(data, shape, GpuContext.Default);

            Assert.IsTrue(data.SequenceEqual(storage.ToArray()));
            Assert.IsTrue(storage.CopiedToHost);
            Assert.IsFalse(storage.CopiedToDevice);

            // Copy to device
            storage.CopyToDevice();
            Assert.IsFalse(storage.CopiedToHost);
            Assert.IsTrue(storage.CopiedToDevice);

            // Copy back to host
            storage.CopyToHost();
            Assert.IsTrue(data.SequenceEqual(storage.ToArray()));
            Assert.IsTrue(storage.CopiedToHost);
            Assert.IsFalse(storage.CopiedToDevice);
        }

        [TestMethod]
        public void ReShape_Data()
        {
            var data = new[] { 1.0, 2.0, 3.0 };
            var volume = new Double.Volume(data, new Shape(3), GpuContext.Default);

            var reshaped = volume.ReShape(1, -1);

            Assert.IsTrue(reshaped.ToArray().SequenceEqual(volume.Storage.ToArray()));
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
    }
}