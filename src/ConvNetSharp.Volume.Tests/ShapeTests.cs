using System;
using NUnit.Framework;

namespace ConvNetSharp.Volume.Tests
{
    [TestFixture]
    public class ShapeTests
    {
        [Test]
        public void GuessUnknownDimension()
        {
            var shape = new Shape(2, -1);
            shape.GuessUnknownDimension(10);

            Assert.AreEqual(5, shape.Dimensions[1]);
        }

        [Test]
        public void IncompatibleTotalLength()
        {
            var shape = new Shape(2, -1);
            Assert.Throws<ArgumentException>(() => shape.GuessUnknownDimension(9));
        }

        [Test]
        public void NegativeTotalLength()
        {
            var shape = new Shape(2, -1);
            Assert.Throws<ArgumentException>(() => shape.GuessUnknownDimension(-1));
        }

        [Test]
        public void SetDimension()
        {
            var shape = new Shape(2, 2);
            Assert.AreEqual(2, shape.Dimensions[0]);
            Assert.AreEqual(2, shape.Dimensions[1]);
            Assert.AreEqual(4, shape.TotalLength);

            shape.SetDimension(0, 1);
            Assert.AreEqual(1, shape.Dimensions[0]);
            Assert.AreEqual(2, shape.Dimensions[1]);
            Assert.AreEqual(2, shape.TotalLength);
        }

        [Test]
        public void ZeroDimension()
        {
            Assert.Throws<ArgumentException>(() => new Shape(2, 0));
        }
    }
}