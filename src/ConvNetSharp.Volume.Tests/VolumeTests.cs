using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.Tests
{
    public abstract class VolumeTests<T> where T : struct, IEquatable<T>, IFormattable
    {
        [TestMethod]
        public void FromScalar()
        {
            T x = (T)Convert.ChangeType(-1.0, typeof(T));
            Volume<T> vol = x;

            Assert.AreEqual(x, vol.ToArray()[0]);
        }

        [TestMethod]
        public void Add1D()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(3));

            left.DoAdd(right, result);
            AssertNumber.AreEqual(2.0, result.Get(0));
            AssertNumber.AreEqual(4.0, result.Get(1));
            AssertNumber.AreEqual(6.0, result.Get(2));
        }

        [TestMethod]
        public void Add2D()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            var right = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            var result = BuilderInstance<T>.Volume.SameAs(left.Shape);

            left.DoAdd(right, result);
            AssertNumber.AreEqual(2.0, result.Get(0, 0));
            AssertNumber.AreEqual(4.0, result.Get(1, 0));
            AssertNumber.AreEqual(6.0, result.Get(0, 1));
            AssertNumber.AreEqual(8.0, result.Get(1, 1));
        }

        [TestMethod]
        public void AddBroadcast()
        {
            var volume = NewVolume(new[]
            {
                1.0, 2.0,
                3.0, 4.0,
                1.0, 2.0,
                3.0, 4.0,
                1.0, 2.0,
                3.0, 4.0
            }, new Shape(2, 2, 3));

            var bias = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(1, 1, 3));
            var result = BuilderInstance<T>.Volume.SameAs(volume.Shape);

            volume.DoAdd(bias, result);
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0));
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 1));
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 2));
        }

        [TestMethod]
        public void BuilderArray()
        {
            var array = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
            var volume = NewVolume(array, new Shape(5));

            AssertNumber.AreEqual(5, volume.Shape.GetDimension(0));
            for (var i = 0; i < 5; i++)
            {
                AssertNumber.AreEqual(array[i], volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderEmpty()
        {
            var example = NewVolume(new[] { 1.0 }, new Shape(1));
            var volume = BuilderInstance<T>.Volume.SameAs(example.Storage, new Shape(10));

            // SameAs creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is empty
            Assert.AreEqual(10, volume.Shape.GetDimension(0));

            for (var i = 0; i < 10; i++)
            {
                AssertNumber.AreEqual(0.0, volume.Get(i));
            }
        }

        [TestMethod]
        public void Negate()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(3));

            x.DoNegate(result);
            AssertNumber.AreEqual(-1.0, result.Get(0));
            AssertNumber.AreEqual(-2.0, result.Get(1));
            AssertNumber.AreEqual(-3.0, result.Get(2));
        }

        protected abstract Volume<T> NewVolume(double[] values, Shape shape);

        [TestMethod]
        public void Shape2D()
        {
            var volume = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            AssertNumber.AreEqual(2, volume.Shape.GetDimension(0));
            AssertNumber.AreEqual(2, volume.Shape.GetDimension(1));
        }


        [TestMethod]
        public void ToArray()
        {
            var doubles = new[] { 1.0, 2.0, 3.0 };
            var v = NewVolume(doubles, new Shape(3));

            var array = v.ToArray();

            Assert.AreNotSame(doubles, array);
            foreach (var pair in doubles.Zip(array, (a, b) => new { a, b }))
            {
                AssertNumber.AreEqual(pair.a, pair.b);
            }
        }
    }
}