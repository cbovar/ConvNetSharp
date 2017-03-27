using System;
using ConvNetSharp.Volume.GPU.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestClass]
    public class GenericVolumeTests
    {
        [ClassInitialize]
        public static void ClassInit(TestContext context)
        {
            BuilderInstance<double>.Volume = new VolumeBuilder();
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