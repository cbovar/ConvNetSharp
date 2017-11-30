using System;
using System.Linq;
using ConvNetSharp.Core;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.Tests
{
    [TestClass]
    public abstract class VolumeTests<T> where T : struct, IEquatable<T>, IFormattable
    {
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
        public void Div1D()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 4.0, 6.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(3));

            left.DoDivide(right, result);
            AssertNumber.AreEqual(0.5, result.Get(0));
            AssertNumber.AreEqual(0.5, result.Get(1));
            AssertNumber.AreEqual(0.5, result.Get(2));
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
        public void BiasBackward()
        {
            var outputGradient = NewVolume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0
                },
                new Shape(2, 1, 3, 1));

            var biasGradient = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 3, 1));

            outputGradient.BiasGradient(biasGradient);

            AssertNumber.AreEqual(3.0, biasGradient.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(4.0, biasGradient.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(5.0, biasGradient.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void BiasBackwardBatch()
        {
            var outputGradient = NewVolume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0,
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0
                },
                new Shape(2, 1, 3, 2));

            var biasGradient = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 3, 1));

            outputGradient.BiasGradient(biasGradient);

            AssertNumber.AreEqual(6.0, biasGradient.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(8.0, biasGradient.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(10.0, biasGradient.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void Builder()
        {
            var example = NewVolume(new[] { 1.0 }, new Shape(1));
            var volume = BuilderInstance<T>.Volume.SameAs(example.Storage, Ops<T>.One, new Shape(10));

            // From creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is filled with provided value
            AssertNumber.AreEqual(10, volume.Shape.GetDimension(0));
            for (var i = 0; i < 10; i++)
            {
                AssertNumber.AreEqual(1.0, volume.Get(i));
            }
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

            // From creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            Assert.AreEqual(10, volume.Shape.GetDimension(0));
            // - is empty
            for (var i = 0; i < 10; i++)
            {
                AssertNumber.AreEqual(0.0, volume.Get(i));
            }
        }

        [TestMethod]
        public void Convolve()
        {
            // 3x3x3x1
            var input = NewVolume(new double[27].Populate(1.0), new Shape(3, 3, 3, 1));

            // 2x2x3x2
            var filter = NewVolume(
                new double[12].Populate(1.0f).Concat(new double[12].Populate(2.0)).ToArray(),
                new Shape(2, 2, 3, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 2, 1));

            input.DoConvolution(filter, 0, 2, result);

            // 1x1x2x1
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(1, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 0));
            AssertNumber.AreEqual(24.0, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void ConvolveBatch()
        {
            // 3x3x3x2
            var input = NewVolume(new double[27 * 2].Populate(1.0), new Shape(3, 3, 3, 2));

            // 2x2x3x2
            var filter = NewVolume(
                new double[12].Populate(1.0f).Concat(new double[12].Populate(2.0)).ToArray(),
                new Shape(2, 2, 3, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 2, 2));

            input.DoConvolution(filter, 0, 2, result);

            // 1x1x2x2
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(2, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(24.0, result.Storage.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(24.0, result.Storage.Get(0, 0, 1, 1));
        }

        [TestMethod]
        public void ConvolveGradient()
        {
            // 3x3x3x1
            var input = NewVolume(new double[27].Populate(1.0), new Shape(3, 3, 3, 1));

            // 2x2x3x2
            var filter = NewVolume(
                new double[12].Populate(1.0).Concat(new double[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2));

            var outputGradient = NewVolume(new[] { 2.0, 3.0 }, new Shape(1, 1, 2, 1));

            var inputGradient = BuilderInstance<T>.Volume.SameAs(input.Storage, input.Shape);
            var filterGradient = BuilderInstance<T>.Volume.SameAs(filter.Storage, filter.Shape);

            input.DoConvolutionGradient(filter, outputGradient, inputGradient, filterGradient, 0, 2);

            AssertNumber.AreEqual(8.0, inputGradient.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 2, 0));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 1, 0));
        }

        [TestMethod]
        public void ConvolveGradientBatch()
        {
            // 3x3x3x2
            var input = NewVolume(new double[27 * 2].Populate(1.0), new Shape(3, 3, 3, 2));

            // 2x2x3x2
            var filter = NewVolume(
                new double[12].Populate(1.0).Concat(new double[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2));

            var outputGradient = NewVolume(new[]
            {
                2.0, 3.0,
                4.0, 5.0
            }, new Shape(1, 1, 2, 2));

            var inputGradient = BuilderInstance<T>.Volume.SameAs(input.Storage, input.Shape);
            var filterGradient = BuilderInstance<T>.Volume.SameAs(filter.Storage, filter.Shape);

            input.DoConvolutionGradient(filter, outputGradient, inputGradient, filterGradient, 0, 2);

            // input gradient
            AssertNumber.AreEqual(8.0, inputGradient.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 2, 0));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 1, 0));

            AssertNumber.AreEqual(14.0, inputGradient.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 2, 1));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 1, 1));

            // filter gradient
            AssertNumber.AreEqual(1.0, filter.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(1.0, filter.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(1.0, filter.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(2.0, filter.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(2.0, filter.Get(0, 0, 1, 1));
            AssertNumber.AreEqual(2.0, filter.Get(0, 0, 2, 1));
        }

        [TestMethod]
        public void DoAddToSame()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            var right = NewVolume(new[] { 0.1, 0.2, 0.3, 0.4 }, new Shape(2, -1));

            right.DoAdd(left, right);

            AssertNumber.AreEqual(1.1, right.Get(0, 0), 1e-5);
            AssertNumber.AreEqual(2.2, right.Get(1, 0), 1e-5);
            AssertNumber.AreEqual(3.3, right.Get(0, 1), 1e-5);
            AssertNumber.AreEqual(4.4, right.Get(1, 1), 1e-5);
        }

        [TestMethod]
        public void DoSubstractFrom()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 0.0, 1.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(left.Shape);

            right.DoSubtractFrom(left, result);

            AssertNumber.AreEqual(-1.0, result.Get(0));
            AssertNumber.AreEqual(2.0, result.Get(1));
            AssertNumber.AreEqual(2.0, result.Get(2));
        }

        [TestMethod]
        public void DoSubstractFromInPlace()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 0.0, 1.0 }, new Shape(3));

            right.DoSubtractFrom(left, left);

            AssertNumber.AreEqual(-1.0, left.Get(0));
            AssertNumber.AreEqual(2.0, left.Get(1));
            AssertNumber.AreEqual(2.0, left.Get(2));
        }

        [TestMethod]
        public void FromScalar()
        {
            var x = (T)Convert.ChangeType(-1.0, typeof(T));
            Volume<T> vol = x;

            Assert.AreEqual(x, vol.ToArray()[0]);
        }

        /// <summary>
        ///     Fully connection can be expressed as a convolution with 1x1 filters
        /// </summary>
        [TestMethod]
        public void FullyCon()
        {
            // 1x3x1x1
            var input = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(1, 1, 3, 1));

            // 1x1x3x2
            var filter = NewVolume(
                new[] { 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 },
                new Shape(1, 1, 3, 2));

            var result = input.Convolve(filter, 0, 1);

            // 1x1x2x1
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(1, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(6.0, result.Storage.Get(0, 0, 0));
            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void Max1D()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1));

            x.DoMax(result);
            AssertNumber.AreEqual(3.0, result.Get(0));
        }

        [TestMethod]
        public void Max2DBatch()
        {
            var x = NewVolume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 4.0,

                    7.0, -20.0,
                    3.0, 4.0
                }, new Shape(2, 2, 1, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 2));

            x.DoMax(result);
            AssertNumber.AreEqual(4.0, result.Get(0));
            AssertNumber.AreEqual(7.0, result.Get(1));
        }

        [TestMethod]
        public void Min1D()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1));

            x.DoMin(result);
            AssertNumber.AreEqual(1.0, result.Get(0));
        }

        [TestMethod]
        public void Min2DBatch()
        {
            var x = NewVolume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 4.0,

                    7.0, -20.0,
                    3.0, 4.0
                }, new Shape(2, 2, 1, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 2));

            x.DoMin(result);
            AssertNumber.AreEqual(1.0, result.Get(0));
            AssertNumber.AreEqual(-20.0, result.Get(1));
        }

        [TestMethod]
        public void Sum1D()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1));

            x.DoSum(result);
            AssertNumber.AreEqual(6.0, result.Get(0));
        }

        [TestMethod]
        public void SumOverBatch()
        {
            var x = NewVolume(new[] {
                1.0,
                3.0,
                5.0}, new Shape(1, 1, 1, 3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 1));

            x.DoSum(result);
            AssertNumber.AreEqual(9.0, result.Get(0));
        }

        [TestMethod]
        public void Sum2DBatch()
        {
            var x = NewVolume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 4.0,

                    7.0, -20.0,
                    3.0, 4.0
                }, new Shape(2, 2, 1, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 2));

            x.DoSum(result);
            AssertNumber.AreEqual(10.0, result.Get(0));
            AssertNumber.AreEqual(-6.0, result.Get(1));
        }

        [TestMethod]
        public void Norm11D()
        {
            var x = NewVolume(new[] { -1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1));

            x.DoNorm1(result);
            AssertNumber.AreEqual(6.0, result.Get(0));
        }

        [TestMethod]
        public void Norm12DBatch()
        {
            var x = NewVolume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 4.0,

                    7.0, -20.0,
                    3.0, 4.0
                }, new Shape(2, 2, 1, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 2));

            x.DoNorm1(result);
            AssertNumber.AreEqual(10.0, result.Get(0));
            AssertNumber.AreEqual(34.0, result.Get(1));
        }

        [TestMethod]
        public void Multiply()
        {
            var matrix = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var a = NewVolume(matrix, Shape.From(2, 2, 2));
            var b = a.Clone();

            const double eps = 0.00001;

            var result = b.Multiply((T)Convert.ChangeType(0.1, typeof(T)));
            Assert.AreNotSame(b, result);
            Assert.AreNotSame(b.Storage, result.Storage);
            for (var i = 0; i < matrix.Length; i++)
            {
                AssertNumber.AreEqual(matrix[i], a.Get(i), eps);
                AssertNumber.AreEqual(matrix[i], b.Get(i), eps);
                AssertNumber.AreEqual(matrix[i] * 0.1, result.Get(i), eps);
            }

            b = result;
            result = a.Clone();
            a.DoMultiply(b, result);
            for (var i = 0; i < matrix.Length; i++)
            {
                AssertNumber.AreEqual(matrix[i], a.Get(i), eps);
                AssertNumber.AreEqual(matrix[i] * 0.1, b.Get(i), eps);
                AssertNumber.AreEqual(matrix[i] * matrix[i] * 0.1, result.Get(i), eps);
            }
        }

        [TestMethod]
        public void MultiplyCommutative()
        {
            var matrix = new[] { 1.0, 2.0, 3.0 };
            var a = NewVolume(matrix, Shape.From(1, 1, 1, 3));
            var b = NewVolume(new[] { 2.0 }, Shape.From(1));
            var result1 = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 3));
            var result2 = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 3));

            a.DoMultiply(b, result1);
            b.DoMultiply(a, result2);

            Assert.IsTrue(result1.ToArray().SequenceEqual(result2.ToArray()));
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
        public void Pool2DBatch()
        {
            var volume = NewVolume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0,

                2.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 2.0, 14.0,
                4.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 8.0, 2.0
            }, new Shape(4, 4, 1, 2));

            var result = volume.Pool(2, 2, 0, 2);

            Assert.AreEqual(2, result.Shape.GetDimension(0));
            Assert.AreEqual(2, result.Shape.GetDimension(1));
            Assert.AreEqual(1, result.Shape.GetDimension(2));
            Assert.AreEqual(2, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(7.0, result.Get(1, 0, 0, 0));
            AssertNumber.AreEqual(2.0, result.Get(0, 1, 0, 0));
            AssertNumber.AreEqual(4.0, result.Get(1, 1, 0, 0));

            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(14.0, result.Get(1, 0, 0, 1));
            AssertNumber.AreEqual(4.0, result.Get(0, 1, 0, 1));
            AssertNumber.AreEqual(8.0, result.Get(1, 1, 0, 1));
        }

        [TestMethod]
        public void Pool2DGradient()
        {
            var inputActivation = NewVolume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0
            }, new Shape(4, 4));

            var outputActivation = inputActivation.Pool(2, 2, 0, 2);

            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(2, 2));

            var result = outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 2);

            AssertNumber.AreEqual(1.0, result.Get(0, 0));
            AssertNumber.AreEqual(1.0, result.Get(3, 1));
            AssertNumber.AreEqual(1.0, result.Get(0, 2));
            AssertNumber.AreEqual(1.0, result.Get(2, 3));
        }

        [TestMethod]
        public void Pool2DGradientBatch()
        {
            var inputActivation = NewVolume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0,

                2.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 2.0, 14.0,
                4.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 8.0, 2.0
            }, new Shape(4, 4, 1, 2));

            var outputActivation = inputActivation.Pool(2, 2, 0, 2);

            var outputActivationGradient = NewVolume(new[]
            {
                1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0
            }, new Shape(2, 2, 1, 2));

            var result = outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 2);

            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(1.0, result.Get(3, 1, 0, 0));
            AssertNumber.AreEqual(1.0, result.Get(0, 2, 0, 0));
            AssertNumber.AreEqual(1.0, result.Get(2, 3, 0, 0));

            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(2.0, result.Get(3, 1, 0, 1));
            AssertNumber.AreEqual(2.0, result.Get(0, 2, 0, 1));
            AssertNumber.AreEqual(2.0, result.Get(2, 3, 0, 1));
        }

        [TestMethod]
        public void Relu()
        {
            var volume = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));

            var result = volume.Relu();
            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(3.0, result.Get(2));
            AssertNumber.AreEqual(5.0, result.Get(3));
        }

        [TestMethod]
        public void ReluGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));

            var result = outputActivation.ReluGradient(inputActivation, outputActivationGradient);

            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(1.0, result.Get(2));
            AssertNumber.AreEqual(1.0, result.Get(3));
        }

        [TestMethod]
        public void LeakyRelu()
        {
            var volume = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));

            var result = volume.LeakyRelu();
            //Adding a delta of 0.005 to account for floating point randomness
            AssertNumber.AreEqual(-0.01, result.Get(0), 0.005);
            AssertNumber.AreEqual(0.0, result.Get(1), 0.005);
            AssertNumber.AreEqual(3.0, result.Get(2), 0.005);
            AssertNumber.AreEqual(5.0, result.Get(3), 0.005);
        }

        [TestMethod]
        public void LeakyReluGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = inputActivation.LeakyRelu();
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));

            var result = outputActivation.LeakyReluGradient(inputActivation, outputActivationGradient);

            AssertNumber.AreEqual(0.01, result.Get(0), 0.005);
            AssertNumber.AreEqual(1.0, result.Get(1), 0.005);
            AssertNumber.AreEqual(1.0, result.Get(2), 0.005);
            AssertNumber.AreEqual(1.0, result.Get(3), 0.005);
        }

        [TestMethod]
        public void Shape2D()
        {
            var volume = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            AssertNumber.AreEqual(2, volume.Shape.GetDimension(0));
            AssertNumber.AreEqual(2, volume.Shape.GetDimension(1));
        }

        [TestMethod]
        public void Sigmoid()
        {
            var volume = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));

            var result = volume.Sigmoid();
            AssertNumber.AreEqual(1.0 / (1.0 + Math.Exp(1.0)), result.Get(0), 1e-5);
            AssertNumber.AreEqual(1.0 / (1.0 + Math.Exp(0.0)), result.Get(1), 1e-5);
            AssertNumber.AreEqual(1.0 / (1.0 + Math.Exp(-3.0)), result.Get(2), 1e-5);
            AssertNumber.AreEqual(1.0 / (1.0 + Math.Exp(-5.0)), result.Get(3), 1e-5);
        }

        [TestMethod]
        public void SigmoidGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));

            var result = outputActivation.SigmoidGradient(inputActivation, outputActivationGradient);

            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(-6.0, result.Get(2));
            AssertNumber.AreEqual(-20.0, result.Get(3));
        }

        [TestMethod]
        public void Softmax()
        {
            var input1 = NewVolume(new[] { 0.0, 0.0, 0.0, 10000.0 }, new Shape(1, 1, -1, 1));
            var softmax1 = input1.Softmax();
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(1.0, softmax1.Get(0, 0, 3, 0));

            var input2 = NewVolume(new[] { 10000.0, 0.0, 0.0, 10000.0 }, new Shape(1, 1, -1, 1));
            var softmax2 = input2.Softmax();
            AssertNumber.AreEqual(0.5, softmax2.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.5, softmax2.Get(0, 0, 3, 0));
        }

        [TestMethod]
        public void SoftmaxBatch()
        {
            var volume1 = NewVolume(new[]
            {
                0.0, 0.0, 0.0, 10000.0,
                0.0, 0.0, 10000.0, 0.0
            }, new Shape(1, 1, -1, 2));
            var softmax1 = volume1.Softmax();

            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(1.0, softmax1.Get(0, 0, 3, 0));

            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 1, 1));
            AssertNumber.AreEqual(1.0, softmax1.Get(0, 0, 2, 1));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 3, 1));
        }

        [TestMethod]
        public void SoftmaxGradient()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = NewVolume(new[] { 1.0, 0.1, 0.1, 0.1 }, new Shape(1, 1, -1, 1));

            // output  = softmax(input)
            var output = input.Softmax();

            // groundTruth = [0, 1, 0 , 0]
            var correctClass = 1;
            var groundTruth = NewVolume(new double[4], new Shape(1, 1, -1, 1));
            groundTruth.Set(0, 0, correctClass, (T)Convert.ChangeType(1.0, typeof(T)));

            var inputGradient = output.SoftmaxGradient(groundTruth);

            AssertNumber.AreEqual(-0.08251689706523138, inputGradient.Get(0, 0, 0, 0), 1e-4);
            AssertNumber.AreEqual(0.14961463059055374, inputGradient.Get(0, 0, 1, 0), 1e-4);
            AssertNumber.AreEqual(-0.033548866762661167, inputGradient.Get(0, 0, 2, 0), 1e-4);
            AssertNumber.AreEqual(-0.033548866762661167, inputGradient.Get(0, 0, 3, 0), 1e-4);
        }

        [TestMethod]
        public void SoftmaxGradientBatch()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = NewVolume(new[]
            {
                1.0, 0.1, 0.1, 0.1,
                0.1, 0.1, 1.0, 0.1
            }, new Shape(1, 1, -1, 2));

            // output  = softmax(input)
            var output = input.Softmax();

            // groundTruth = [ [0, 1, 0 , 0], [0, 0, 0, 1] ]
            var groundTruth = NewVolume(new[]
            {
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            }, new Shape(1, 1, -1, 2));

            var inputGradient = output.SoftmaxGradient(groundTruth);

            AssertNumber.AreEqual(-0.082516897065231382, inputGradient.Get(0, 0, 0, 0), 1e-6);
            AssertNumber.AreEqual(0.14961463059055374, inputGradient.Get(0, 0, 1, 0), 1e-6);
            AssertNumber.AreEqual(-0.033548866762661167, inputGradient.Get(0, 0, 2, 0), 1e-6);
            AssertNumber.AreEqual(-0.033548866762661167, inputGradient.Get(0, 0, 3, 0), 1e-6);
            AssertNumber.AreEqual(-0.033548866762661167, inputGradient.Get(0, 0, 0, 1), 1e-6);
            AssertNumber.AreEqual(-0.033548866762661167, inputGradient.Get(0, 0, 1, 1), 1e-6);
            AssertNumber.AreEqual(-0.082516897065231382, inputGradient.Get(0, 0, 2, 1), 1e-6);
            AssertNumber.AreEqual(0.14961463059055374, inputGradient.Get(0, 0, 3, 1), 1e-6);
        }

        [TestMethod]
        public void SubstractFrom()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 0.0, 1.0 }, new Shape(3));

            var result = left - right;
            AssertNumber.AreEqual(-1.0, result.Get(0));
            AssertNumber.AreEqual(2.0, result.Get(1));
            AssertNumber.AreEqual(2.0, result.Get(2));
        }

        [TestMethod]
        public void Log()
        {
            var volume = NewVolume(new[] { 1, 1.5, 3.0, 5.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            volume.DoLog(result);
            AssertNumber.AreEqual(Math.Log(1.0), result.Get(0), 1e-6);
            AssertNumber.AreEqual(Math.Log(1.5), result.Get(1), 1e-6);
            AssertNumber.AreEqual(Math.Log(3.0), result.Get(2), 1e-6);
            AssertNumber.AreEqual(Math.Log(5.0), result.Get(3), 1e-6);
        }

        [TestMethod]
        public void Exp()
        {
            var volume = NewVolume(new[] { 1, 1.5, 3.0, 5.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            volume.DoExp(result);
            AssertNumber.AreEqual(Math.Exp(1.0), result.Get(0), 1e-5);
            AssertNumber.AreEqual(Math.Exp(1.5), result.Get(1), 1e-5);
            AssertNumber.AreEqual(Math.Exp(3.0), result.Get(2), 1e-5);
            AssertNumber.AreEqual(Math.Exp(5.0), result.Get(3), 1e-5);
        }

        [TestMethod]
        public void Tanh()
        {
            var volume = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));

            var result = volume.Tanh();
            AssertNumber.AreEqual(Math.Tanh(-1.0), result.Get(0), 1e-6);
            AssertNumber.AreEqual(Math.Tanh(0.0), result.Get(1), 1e-6);
            AssertNumber.AreEqual(Math.Tanh(3.0), result.Get(2), 1e-6);
            AssertNumber.AreEqual(Math.Tanh(5.0), result.Get(3), 1e-6);
        }

        [TestMethod]
        public void TanhGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));

            var result = outputActivation.TanhGradient(inputActivation, outputActivationGradient);

            AssertNumber.AreEqual(1.0, result.Get(0), 1e-6);
            AssertNumber.AreEqual(1.0, result.Get(1), 1e-6);
            AssertNumber.AreEqual(-8.0, result.Get(2), 1e-6);
            AssertNumber.AreEqual(-24.0, result.Get(3), 1e-6);
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

        [TestMethod]
        public void Dropout()
        {
            var volume = NewVolume(new double[100].Populate(1.0), new Shape(100));
            var result = NewVolume(new double[100], new Shape(100));

            var dropProb = 0.5;
            volume.DoDropout(result, true, (T)Convert.ChangeType(dropProb, typeof(T)));

            var array = result.Storage.ToArray();
            var c = array.Count(o => o.Equals(Ops<T>.Zero));
            Assert.IsTrue(c > 0);

            var nonZeroEntry = array.First(o => !o.Equals(Ops<T>.Zero));
            AssertNumber.AreEqual(1.0 / (1 - dropProb), nonZeroEntry, 1e-6);

            var inputGradient = BuilderInstance<T>.Volume.SameAs(volume.Storage, volume.Shape);
            var gradient = 1.0;
            var outputActivationGradient = NewVolume(new double[100].Populate(gradient), new Shape(100));
            volume.DoDropoutGradient(volume, outputActivationGradient, inputGradient, (T)Convert.ChangeType(dropProb, typeof(T)));

            array = inputGradient.Storage.ToArray();
            nonZeroEntry = array.First(o => !o.Equals(Ops<T>.Zero));
            AssertNumber.AreEqual(gradient / (1 - dropProb), nonZeroEntry, 1e-6);
        }
    }
}