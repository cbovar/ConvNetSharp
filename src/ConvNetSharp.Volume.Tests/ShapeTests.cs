using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.Tests
{
    [TestClass]
    public class ShapeTests
    {
        [TestMethod]
        public void GuessUnknownDimension()
        {
            var shape = new Shape(2, -1);
            shape.GuessUnkownDimension(10);

            Assert.AreEqual(5, shape.GetDimension(1));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException),
            "Input to reshape is a tensor with 9 values, but the requested shape requires a multiple of 2")]
        public void IncompatibleTotalLength()
        {
            var shape = new Shape(2, -1);
            shape.GuessUnkownDimension(9);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException), "totalLength must be non-negative, not -1")]
        public void NegativeTotalLength()
        {
            var shape = new Shape(2, -1);
            shape.GuessUnkownDimension(-1);
        }

        [TestMethod]
        public void SetDimension()
        {
            var shape = new Shape(2, 2);
            Assert.AreEqual(2, shape.GetDimension(0));
            Assert.AreEqual(2, shape.GetDimension(1));
            Assert.AreEqual(4, shape.TotalLength);

            shape.SetDimension(0, 1);
            Assert.AreEqual(1, shape.GetDimension(0));
            Assert.AreEqual(2, shape.GetDimension(1));
            Assert.AreEqual(2, shape.TotalLength);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException), "Dimension #1 must be non-negative, not 0")]
        public void ZeroDimension()
        {
            var shape = new Shape(2, 0);
            shape.GuessUnkownDimension(10);
        }
    }
}