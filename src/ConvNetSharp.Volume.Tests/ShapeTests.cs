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

        [TestCase(new[] { 2 }, 1, 1, 2, 1)]
        [TestCase(new[] { 5,5 }, 5, 5, 1, 1)]
        [TestCase(new[] { 5, 5, 3 }, 5, 5, 3, 1)]
        [TestCase(new[] { 5, 5, 3, 10 }, 5, 5, 3, 10)]
        public void From_WHCN(int[] dimensions, int dim0, int dim1, int dim2, int dim3)
        {
            var shape = Shape.From(dimensions);

            Assert.AreEqual(dim0, shape.Dimensions[0]);
            Assert.AreEqual(dim1, shape.Dimensions[1]);
            Assert.AreEqual(dim2, shape.Dimensions[2]);
            Assert.AreEqual(dim3, shape.Dimensions[3]);
        }

        [TestCase(new[] { 2 }, 1, 1, 2, 1)]
        [TestCase(new[] { 5, 5 }, 5, 5, 1, 1)]
        [TestCase(new[] { 3, 5, 5 }, 5, 5, 3, 1)]
        [TestCase(new[] { 10, 3, 5, 5 }, 5, 5, 3, 10)]
        public void From_NCWH(int[] dimensions, int dim0, int dim1, int dim2, int dim3)
        {
            var shape = Shape.From(dimensions, Shape.DimensionOrder.NCWH);

            Assert.AreEqual(dim0, shape.Dimensions[0]);
            Assert.AreEqual(dim1, shape.Dimensions[1]);
            Assert.AreEqual(dim2, shape.Dimensions[2]);
            Assert.AreEqual(dim3, shape.Dimensions[3]);
        }
    }
}