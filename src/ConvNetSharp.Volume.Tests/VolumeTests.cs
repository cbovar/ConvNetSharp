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

            left.Add(right, result);
            AssertNumber.AreEqual(2.0, result.Get(0));
            AssertNumber.AreEqual(4.0, result.Get(1));
            AssertNumber.AreEqual(6.0, result.Get(2));
        }

        /// <summary>
        /// Test +=
        /// </summary>
        [TestMethod]
        public void Add1DInPlace()
        {
            var input = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var other = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));

            other.Add(input);
            AssertNumber.AreEqual(2.0, input.Get(0));
            AssertNumber.AreEqual(4.0, input.Get(1));
            AssertNumber.AreEqual(6.0, input.Get(2));
        }

        [TestMethod]
        public void Div1D()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 4.0, 6.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(3));

            left.Divide(right, result);
            AssertNumber.AreEqual(0.5, result.Get(0));
            AssertNumber.AreEqual(0.5, result.Get(1));
            AssertNumber.AreEqual(0.5, result.Get(2));
        }

        [TestMethod]
        public void Div1DInPlace()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 8.0, 10.0 }, new Shape(3));

            left.Divide(right, left);
            AssertNumber.AreEqual(0.5, left.Get(0), 1e-6);
            AssertNumber.AreEqual(0.25, left.Get(1), 1e-6);
            AssertNumber.AreEqual(0.3, left.Get(2), 1e-6);
        }

        [TestMethod]
        public void Add2D()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            var right = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            var result = BuilderInstance<T>.Volume.SameAs(left.Shape);

            left.Add(right, result);
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

            volume.Add(bias, result);
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0));
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 1));
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 2));
        }

        [TestMethod]
        public void AddBroadcastScalar()
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

            var bias = NewVolume(new[] { 1.0, }, new Shape(1));
            var result = BuilderInstance<T>.Volume.SameAs(volume.Shape);

            volume.Add(bias, result);
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0));
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 1));
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 2));
        }

        [TestMethod]
        public void AddBroadcastNeg()
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

            var bias = NewVolume(new[] { -1.0, -2.0, -3.0 }, new Shape(1, 1, 3));
            var result = BuilderInstance<T>.Volume.SameAs(volume.Shape);

            volume.Add(bias, result);
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 0));
            AssertNumber.AreEqual(-1.0, result.Get(0, 0, 1));
            AssertNumber.AreEqual(-2.0, result.Get(0, 0, 2));
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
            var volume = BuilderInstance<T>.Volume.SameAs(example.Storage, Ops<T>.One, new Shape(10)); // shape [1,1,10,1]

            // From creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is filled with provided value
            AssertNumber.AreEqual(10, volume.Shape.Dimensions[2]);
            for (var i = 0; i < 10; i++)
            {
                AssertNumber.AreEqual(1.0, volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderArray()
        {
            var array = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }; // shape [1,1,5,1]
            var volume = NewVolume(array, new Shape(5));

            AssertNumber.AreEqual(5, volume.Shape.Dimensions[2]);
            for (var i = 0; i < 5; i++)
            {
                AssertNumber.AreEqual(array[i], volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderEmpty()
        {
            var example = NewVolume(new[] { 1.0 }, new Shape(1)); ; // shape [1,1,1,1]
            var volume = BuilderInstance<T>.Volume.SameAs(example.Storage, new Shape(10));

            // From creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            Assert.AreEqual(10, volume.Shape.Dimensions[2]);
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

            input.Convolution(filter, 0, 2, result);

            // 1x1x2x1
            Assert.AreEqual(1, result.Shape.Dimensions[0]);
            Assert.AreEqual(1, result.Shape.Dimensions[1]);
            Assert.AreEqual(2, result.Shape.Dimensions[2]);
            Assert.AreEqual(1, result.Shape.Dimensions[3]);

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

            input.Convolution(filter, 0, 2, result);

            // 1x1x2x2
            Assert.AreEqual(1, result.Shape.Dimensions[0]);
            Assert.AreEqual(1, result.Shape.Dimensions[1]);
            Assert.AreEqual(2, result.Shape.Dimensions[2]);
            Assert.AreEqual(2, result.Shape.Dimensions[3]);

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

            input.ConvolutionGradient(filter, outputGradient, filterGradient, 0, 2, inputGradient);

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

            input.ConvolutionGradient(filter, outputGradient, filterGradient, 0, 2, inputGradient);

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

            right.Add(left, right);

            AssertNumber.AreEqual(1.1, right.Get(0, 0), 1e-5);
            AssertNumber.AreEqual(2.2, right.Get(1, 0), 1e-5);
            AssertNumber.AreEqual(3.3, right.Get(0, 1), 1e-5);
            AssertNumber.AreEqual(4.4, right.Get(1, 1), 1e-5);
        }

        [TestMethod]
        public void DoConcat()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, 2, 1, 1));
            var right = NewVolume(new[] { 5.0, 6.0, 7.0 }, new Shape(3, 1, 1, 1));
            var result = NewVolume(new double[7], new Shape(1, 1, 7, 1));
            left.Concat(right, result);

            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 3, 0));
            AssertNumber.AreEqual(5.0, result.Get(0, 0, 4, 0));
            AssertNumber.AreEqual(6.0, result.Get(0, 0, 5, 0));
            AssertNumber.AreEqual(7.0, result.Get(0, 0, 6, 0));
        }

        [TestMethod]
        public void DoConcatExtract()
        {
            var left = NewVolume(new[]
            {
                1.0, 2.0, 3.0, 4.0,
                1.0, 2.0, 3.0, 4.0
            }, new Shape(1, 1, 4, 2));

            var right = NewVolume(new[] { 0.0 }, new Shape(1, 1, 1, 1));
            var concatened = NewVolume(new double[10], new Shape(1, 1, 5, 2));
            right.Concat(left, concatened);

            var extracted = NewVolume(new double[8], new Shape(1, 1, 4, 2));
            concatened.Extract(4, 1, extracted);

            Assert.AreEqual(left.Shape, extracted.Shape);

            for (int i = 0; i < left.Shape.TotalLength; i++)
            {
                Assert.AreEqual(left.Get(i), extracted.Get(i));
            }
        }

        [TestMethod]
        public void DoConcatBroadcast()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(1, 1, 1, 4));
            var right = NewVolume(new[] { 1.0 }, new Shape(1));

            var result = NewVolume(new double[8], new Shape(1, 1, 2, 4));

            left.Concat(right, result);

            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(1.0, result.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(1.0, result.Get(0, 0, 1, 1));
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 0, 2));
            AssertNumber.AreEqual(1.0, result.Get(0, 0, 1, 2));
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 0, 3));
            AssertNumber.AreEqual(1.0, result.Get(0, 0, 1, 3));
        }

        [TestMethod]
        public void DoExtract()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }, new Shape(1, 1, 7, 1));

            var result = NewVolume(new double[4], new Shape(1, 1, 4, 1));
            x.Extract(4, 0, result);

            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 3, 0));

            result = NewVolume(new double[3], new Shape(1, 1, 3, 1));
            x.Extract(3, 4, result);

            AssertNumber.AreEqual(5.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(6.0, result.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(7.0, result.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void DoExtractBatch()
        {
            var x = NewVolume(new[]
            {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5
            }, new Shape(1, 1, 7, 2));

            var result = NewVolume(new double[8], new Shape(1, 1, 4, 2));
            x.Extract(4, 0, result);

            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 3, 0));
            AssertNumber.AreEqual(1.5, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(2.5, result.Get(0, 0, 1, 1));
            AssertNumber.AreEqual(3.5, result.Get(0, 0, 2, 1));
            AssertNumber.AreEqual(4.5, result.Get(0, 0, 3, 1));

            result = NewVolume(new double[6], new Shape(1, 1, 3, 2));
            x.Extract(3, 4, result);

            AssertNumber.AreEqual(5.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(6.0, result.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(7.0, result.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(5.5, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(6.5, result.Get(0, 0, 1, 1));
            AssertNumber.AreEqual(7.5, result.Get(0, 0, 2, 1));
        }

        [TestMethod]
        public void DoSubstractFrom()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 0.0, 1.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(left.Shape);

            right.SubtractFrom(left, result);

            AssertNumber.AreEqual(-1.0, result.Get(0));
            AssertNumber.AreEqual(2.0, result.Get(1));
            AssertNumber.AreEqual(2.0, result.Get(2));
        }

        [TestMethod]
        public void DoSubstractFromInPlace()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 0.0, 1.0 }, new Shape(3));

            right.SubtractFrom(left, left);

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

            var outputShape = Volume<T>.ComputeConvolutionShape(input.Shape, filter.Shape, 0, 1);
            var result = NewVolume(new double[outputShape.TotalLength], outputShape);
            input.Convolution(filter, 0, 1, result);

            // 1x1x2x1
            Assert.AreEqual(1, result.Shape.Dimensions[0]);
            Assert.AreEqual(1, result.Shape.Dimensions[1]);
            Assert.AreEqual(2, result.Shape.Dimensions[2]);
            Assert.AreEqual(1, result.Shape.Dimensions[3]);

            AssertNumber.AreEqual(6.0, result.Storage.Get(0, 0, 0));
            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void Max1D()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1));

            x.Max(result);
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

            x.Max(result);
            AssertNumber.AreEqual(4.0, result.Get(0));
            AssertNumber.AreEqual(7.0, result.Get(1));
        }

        [TestMethod]
        public void Min1D()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1));

            x.Min(result);
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

            x.Min(result);
            AssertNumber.AreEqual(1.0, result.Get(0));
            AssertNumber.AreEqual(-20.0, result.Get(1));
        }

        [TestMethod]
        public void Sum1D()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1));

            x.Sum(result);
            AssertNumber.AreEqual(6.0, result.Get(0));
        }

        [TestMethod]
        public void Tile1D()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var reps = NewVolume(new[] { 2.0 }, new Shape(1));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(6));

            x.Tile(reps, result);
            AssertNumber.AreEqual(1.0, result.Get(0));
            AssertNumber.AreEqual(2.0, result.Get(1));
            AssertNumber.AreEqual(3.0, result.Get(2));
            AssertNumber.AreEqual(1.0, result.Get(3));
            AssertNumber.AreEqual(2.0, result.Get(4));
            AssertNumber.AreEqual(3.0, result.Get(5));
        }

        [TestMethod]
        public void TileScalar()
        {
            var x = NewVolume(new[] { 1.0 }, new Shape(1));
            var reps = NewVolume(new[] { 1.0, 1.0, 1.0, 50.0 }, new Shape(4));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 50));

            x.Tile(reps, result);

            for (int i = 0; i < 50; i++)
            {
                AssertNumber.AreEqual(1.0, result.Get(i));
            }
        }

        [TestMethod]
        public void Tile2D()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3, 1, 1, 1));
            var reps = NewVolume(new[] { 2.0, 2.0 }, new Shape(2, 1, 1, 1));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(6, 2));

            x.Tile(reps, result);

            AssertNumber.AreEqual(1.0, result.Get(0, 0));
            AssertNumber.AreEqual(2.0, result.Get(1, 0));
            AssertNumber.AreEqual(3.0, result.Get(2, 0));
            AssertNumber.AreEqual(1.0, result.Get(3, 0));
            AssertNumber.AreEqual(2.0, result.Get(4, 0));
            AssertNumber.AreEqual(3.0, result.Get(5, 0));

            AssertNumber.AreEqual(1.0, result.Get(0, 1));
            AssertNumber.AreEqual(2.0, result.Get(1, 1));
            AssertNumber.AreEqual(3.0, result.Get(2, 1));
            AssertNumber.AreEqual(1.0, result.Get(3, 1));
            AssertNumber.AreEqual(2.0, result.Get(4, 1));
            AssertNumber.AreEqual(3.0, result.Get(5, 1));
        }

        [TestMethod]
        public void SumOverBatch()
        {
            var x = NewVolume(new[] {
                1.0,
                3.0,
                5.0}, new Shape(1, 1, 1, 3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 1, 1));

            x.Sum(result);
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

            x.Sum(result);
            AssertNumber.AreEqual(10.0, result.Get(0));
            AssertNumber.AreEqual(-6.0, result.Get(1));
        }

        [TestMethod]
        public void Norm11D()
        {
            var x = NewVolume(new[] { -1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1));

            x.Norm1(result);
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

            x.Norm1(result);
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

            var result = NewVolume(new double[8], Shape.From(2, 2, 2));
            b.Multiply((T)Convert.ChangeType(0.1, typeof(T)), result);
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
            a.Multiply(b, result);
            for (var i = 0; i < matrix.Length; i++)
            {
                AssertNumber.AreEqual(matrix[i], a.Get(i), eps);
                AssertNumber.AreEqual(matrix[i] * 0.1, b.Get(i), eps);
                AssertNumber.AreEqual(matrix[i] * matrix[i] * 0.1, result.Get(i), eps);
            }
        }

        /// <summary>
        /// Test multiplication by a factor using input volume as output. (i.e. volume *= factor)
        /// </summary>
        [TestMethod]
        public void MultiplyByFactor()
        {
            var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var volume = NewVolume(data, Shape.From(2, 2, 2));
            volume.Multiply((T)Convert.ChangeType(0.1, typeof(T)), volume); // volume *= 0.1

            for (var i = 0; i < data.Length; i++)
            {
                AssertNumber.AreEqual(data[i] * 0.1, volume.Get(i), 0.00001);
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

            a.Multiply(b, result1);
            b.Multiply(a, result2);

            Assert.IsTrue(result1.ToArray().SequenceEqual(result2.ToArray()));
        }

        [TestMethod]
        public void MatMultiplyByVector()
        {
            var matrixA = new[] { 1.0, 2.0, 3.0, 4.0 };
            var a = NewVolume(matrixA, Shape.From(2, 2));

            var matrixB = new[] { 1.0, 2.0 };
            var b = NewVolume(matrixB, Shape.From(1, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 2));

            a.MatMultiply(b, result);

            AssertNumber.AreEqual(5.0, result.Get(0, 0));
            AssertNumber.AreEqual(11.0, result.Get(0, 1));
        }

        [TestMethod]
        public void MatMultiply()
        {
            var matrixA = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var a = NewVolume(matrixA, Shape.From(2, 4));

            var matrixB = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
            var b = NewVolume(matrixB, Shape.From(3, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(3, 4));

            a.MatMultiply(b, result);

            AssertNumber.AreEqual(9.0, result.Get(0, 0));
            AssertNumber.AreEqual(12.0, result.Get(1, 0));
            AssertNumber.AreEqual(15.0, result.Get(2, 0));

            AssertNumber.AreEqual(19.0, result.Get(0, 1));
            AssertNumber.AreEqual(26.0, result.Get(1, 1));
            AssertNumber.AreEqual(33.0, result.Get(2, 1));

            AssertNumber.AreEqual(29.0, result.Get(0, 2));
            AssertNumber.AreEqual(40.0, result.Get(1, 2));
            AssertNumber.AreEqual(51.0, result.Get(2, 2));

            AssertNumber.AreEqual(39.0, result.Get(0, 3));
            AssertNumber.AreEqual(54.0, result.Get(1, 3));
            AssertNumber.AreEqual(69.0, result.Get(2, 3));
        }

        [TestMethod]
        public void MatMultiplyBatch()
        {
            var matrixA = new[]
            {
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                3.0, 3.0, 3.0, 4.0, 4.0, 4.0
            };
            var a = NewVolume(matrixA, Shape.From(3, 2, 1, 2));

            var matrixB = new[]
            {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            };
            var b = NewVolume(matrixB, Shape.From(2, 3, 1, 2));

            var result = BuilderInstance<T>.Volume.SameAs(new Shape(2, 2, 1, 2));

            a.MatMultiply(b, result);

            AssertNumber.AreEqual(9.0, result.Get(0, 0));
            AssertNumber.AreEqual(12.0, result.Get(1, 0));
            AssertNumber.AreEqual(18.0, result.Get(0, 1));
            AssertNumber.AreEqual(24.0, result.Get(1, 1));

            AssertNumber.AreEqual(27.0, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(36.0, result.Get(1, 0, 0, 1));
            AssertNumber.AreEqual(36.0, result.Get(0, 1, 0, 1));
            AssertNumber.AreEqual(48.0, result.Get(1, 1, 0, 1));
        }

        [TestMethod]
        public void Negate()
        {
            var x = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(new Shape(3));

            x.Negate(result);
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

            var outputShape = Volume<T>.ComputePoolShape(volume.Shape, 2, 2, 0, 0, 2, 2);
            var result = NewVolume(new double[outputShape.TotalLength], outputShape);
            volume.Pool(2, 2, 0, 0, 2, 2, result);

            Assert.AreEqual(2, result.Shape.Dimensions[0]);
            Assert.AreEqual(2, result.Shape.Dimensions[1]);
            Assert.AreEqual(1, result.Shape.Dimensions[2]);
            Assert.AreEqual(2, result.Shape.Dimensions[3]);

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

            var outputShape = Volume<T>.ComputePoolShape(inputActivation.Shape, 2, 2, 0, 0, 2, 2);
            var outputActivation = NewVolume(new double[outputShape.TotalLength], outputShape);
            inputActivation.Pool(2, 2, 0, 0, 2, 2, outputActivation);

            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(2, 2));

            var result = NewVolume(new double[16], new Shape(4, 4));
            outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 0, 2, 2, result);

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

            var outputShape = Volume<T>.ComputePoolShape(inputActivation.Shape, 2, 2, 0, 0, 2, 2);
            var outputActivation = NewVolume(new double[outputShape.TotalLength], outputShape);
            inputActivation.Pool(2, 2, 0, 0, 2, 2, outputActivation);

            var outputActivationGradient = NewVolume(new[]
            {
                1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0
            }, new Shape(2, 2, 1, 2));


            var result = NewVolume(new double[32], new Shape(4, 4, 1, 2));
            outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 0, 2, 2, result);

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
            var result = NewVolume(new double[4], new Shape(4));

            volume.Relu(result);

            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(3.0, result.Get(2));
            AssertNumber.AreEqual(5.0, result.Get(3));
        }

        /// <summary>
        /// Relu and LearkyRelu with alpha = 0 should return the same result
        /// </summary>
        [TestMethod]
        public void ReluAndLeakyRelu()
        {
            var volume = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));

            var reluResult = NewVolume(new double[4], new Shape(4));
            volume.Relu(reluResult);

            var leakyReluResult = NewVolume(new double[4], new Shape(4));
            volume.LeakyRelu((T)Convert.ChangeType(0.0, typeof(T)), leakyReluResult);

            Assert.IsTrue(reluResult.ToArray().SequenceEqual(leakyReluResult.ToArray()));
        }

        [TestMethod]
        public void ReluGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = NewVolume(new double[4], new Shape(4));

            inputActivation.Relu(outputActivation);

            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));

            var result = NewVolume(new double[4], new Shape(4));
            outputActivation.ReluGradient(inputActivation, outputActivationGradient, result);

            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(1.0, result.Get(2));
            AssertNumber.AreEqual(1.0, result.Get(3));
        }

        [TestMethod]
        public void LeakyRelu()
        {
            var volume = NewVolume(new[]
            {
                -1.0, 0.0, 3.0, 5.0,
                6.0, 4.0, 0.0, -5.0
            }, new Shape(1, 1, 4, 2));

            var result = NewVolume(new double[8], new Shape(1, 1, 4, 2));

            var alpha = (T)Convert.ChangeType(0.01, typeof(T));
            volume.LeakyRelu(alpha, result);
            //Adding a delta of 0.005 to account for floating point randomness
            AssertNumber.AreEqual(-0.01, result.Get(0, 0, 0, 0), 0.005);
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 1, 0), 0.005);
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 2, 0), 0.005);
            AssertNumber.AreEqual(5.0, result.Get(0, 0, 3, 0), 0.005);

            AssertNumber.AreEqual(6.0, result.Get(0, 0, 0, 1), 0.005);
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 1, 1), 0.005);
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 2, 1), 0.005);
            AssertNumber.AreEqual(-0.05, result.Get(0, 0, 3, 1), 0.005);
        }

        [TestMethod]
        public void LeakyReluGradient()
        {
            var inputActivation = NewVolume(new[]
            {
                -1.0, 0.0, 3.0, 5.0,
                6.0, 4.0, 0.0, -5.0
            }, new Shape(1, 1, 4, 2));

            var alpha = (T)Convert.ChangeType(0.01, typeof(T));
            var outputActivation = NewVolume(new double[8], new Shape(1, 1, 4, 2));
            inputActivation.LeakyRelu(alpha, outputActivation);

            var outputActivationGradient = NewVolume(new[]
            {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0
            }, new Shape(1, 1, 4, 2));

            var result = NewVolume(new double[8], new Shape(1, 1, 4, 2));
            outputActivation.LeakyReluGradient(outputActivationGradient, result, alpha);

            AssertNumber.AreEqual(0.01, result.Get(0, 0, 0, 0), 0.005);
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 1, 0), 0.005);
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 2, 0), 0.005);
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 3, 0), 0.005);

            AssertNumber.AreEqual(5.0, result.Get(0, 0, 0, 1), 0.005);
            AssertNumber.AreEqual(6.0, result.Get(0, 0, 1, 1), 0.005);
            AssertNumber.AreEqual(7.0, result.Get(0, 0, 2, 1), 0.005);
            AssertNumber.AreEqual(0.08, result.Get(0, 0, 3, 1), 0.005);
        }

        [TestMethod]
        public void Shape2D()
        {
            var volume = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            AssertNumber.AreEqual(2, volume.Shape.Dimensions[0]);
            AssertNumber.AreEqual(2, volume.Shape.Dimensions[1]);
        }

        [TestMethod]
        public void Sigmoid()
        {
            var volume = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            volume.Sigmoid(result);

            AssertNumber.AreEqual(1.0 / (1.0 + Math.Exp(1.0)), result.Get(0), 1e-5);
            AssertNumber.AreEqual(1.0 / (1.0 + Math.Exp(0.0)), result.Get(1), 1e-5);
            AssertNumber.AreEqual(1.0 / (1.0 + Math.Exp(-3.0)), result.Get(2), 1e-5);
            AssertNumber.AreEqual(1.0 / (1.0 + Math.Exp(-5.0)), result.Get(3), 1e-5);
        }

        [TestMethod]
        public void SigmoidGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = NewVolume(new double[4], new Shape(4));
            inputActivation.Relu(outputActivation);
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            outputActivation.SigmoidGradient(inputActivation, outputActivationGradient, result);

            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(-6.0, result.Get(2));
            AssertNumber.AreEqual(-20.0, result.Get(3));
        }

        [TestMethod]
        public void Softmax()
        {
            var input1 = NewVolume(new[] { 0.0, 0.0, 0.0, 10000.0 }, new Shape(1, 1, -1, 1));
            var result = NewVolume(new double[4], new Shape(4));

            input1.Softmax(result);

            AssertNumber.AreEqual(0.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(1.0, result.Get(0, 0, 3, 0));

            var input2 = NewVolume(new[] { 10000.0, 0.0, 0.0, 10000.0 }, new Shape(1, 1, -1, 1));
            input2.Softmax(result);

            AssertNumber.AreEqual(0.5, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.5, result.Get(0, 0, 3, 0));
        }

        [TestMethod]
        public void SoftmaxBatch()
        {
            var volume1 = NewVolume(new[]
            {
                0.0, 0.0, 0.0, 10000.0,
                0.0, 0.0, 10000.0, 0.0
            }, new Shape(1, 1, -1, 2));
            var result = NewVolume(new double[8], new Shape(1, 1, 4, 2));

            volume1.Softmax(result);

            AssertNumber.AreEqual(0.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(1.0, result.Get(0, 0, 3, 0));

            AssertNumber.AreEqual(0.0, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 1, 1));
            AssertNumber.AreEqual(1.0, result.Get(0, 0, 2, 1));
            AssertNumber.AreEqual(0.0, result.Get(0, 0, 3, 1));
        }

        [TestMethod]
        public void SoftmaxGradient()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = NewVolume(new[] { 1.0, 0.1, 0.1, 0.1 }, new Shape(1, 1, -1, 1));
            var output = NewVolume(new double[4], new Shape(4));

            // output = softmax(input)
            input.Softmax(output);

            // groundTruth = [0, 1, 0 , 0]
            var correctClass = 1;
            var groundTruth = NewVolume(new double[4], new Shape(1, 1, -1, 1));
            groundTruth.Set(0, 0, correctClass, (T)Convert.ChangeType(1.0, typeof(T)));

            var inputGradient = NewVolume(new double[4], new Shape(4));
            output.SoftmaxGradient(groundTruth, inputGradient);

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
            var output = NewVolume(new double[8], new Shape(1, 1, 4, 2));

            // output  = softmax(input)
            input.Softmax(output);

            // groundTruth = [ [0, 1, 0 , 0], [0, 0, 0, 1] ]
            var groundTruth = NewVolume(new[]
            {
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            }, new Shape(1, 1, -1, 2));

            var inputGradient = NewVolume(new double[8], new Shape(1, 1, 4, 2));
            output.SoftmaxGradient(groundTruth, inputGradient);

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

            var result = NewVolume(new double[3], new Shape(3));
            right.SubtractFrom(left, result);

            AssertNumber.AreEqual(-1.0, result.Get(0));
            AssertNumber.AreEqual(2.0, result.Get(1));
            AssertNumber.AreEqual(2.0, result.Get(2));
        }

        [TestMethod]
        public void Log()
        {
            var volume = NewVolume(new[] { 1, 1.5, 3.0, 5.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            volume.Log(result);
            AssertNumber.AreEqual(Math.Log(1.0), result.Get(0), 1e-6);
            AssertNumber.AreEqual(Math.Log(1.5), result.Get(1), 1e-6);
            AssertNumber.AreEqual(Math.Log(3.0), result.Get(2), 1e-6);
            AssertNumber.AreEqual(Math.Log(5.0), result.Get(3), 1e-6);
        }

        [TestMethod]
        public void Power()
        {
            var u = NewVolume(new[] { 1, 1.5, 3.0, 5.0 }, new Shape(4));
            var v = NewVolume(new[] { 0, 2.0, 10.0, -0.5 }, new Shape(4));

            var result = NewVolume(new double[4], new Shape(4));

            u.Power(v, result);

            AssertNumber.AreEqual(Math.Pow(1.0, 0.0), result.Get(0), 1e-5);
            AssertNumber.AreEqual(Math.Pow(1.5, 2.0), result.Get(1), 1e-5);
            AssertNumber.AreEqual(Math.Pow(3.0, 10.0), result.Get(2), 1e-5);
            AssertNumber.AreEqual(Math.Pow(5.0, -0.5), result.Get(3), 1e-5);
        }

        [TestMethod]
        public void Power2()
        {
            var volume = NewVolume(new[] { 1, 1.5, 3.0, 5.0 }, new Shape(4));
            var two = NewVolume(new[] { 2.0 }, new Shape(1));

            volume.Power(two, volume);

            AssertNumber.AreEqual(Math.Pow(1.0, 2.0), volume.Get(0), 1e-5);
            AssertNumber.AreEqual(Math.Pow(1.5, 2.0), volume.Get(1), 1e-5);
            AssertNumber.AreEqual(Math.Pow(3.0, 2.0), volume.Get(2), 1e-5);
            AssertNumber.AreEqual(Math.Pow(5.0, 2.0), volume.Get(3), 1e-5);
        }

        [TestMethod]
        public void Exp()
        {
            var volume = NewVolume(new[] { 1, 1.5, 3.0, 5.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            volume.Exp(result);
            AssertNumber.AreEqual(Math.Exp(1.0), result.Get(0), 1e-5);
            AssertNumber.AreEqual(Math.Exp(1.5), result.Get(1), 1e-5);
            AssertNumber.AreEqual(Math.Exp(3.0), result.Get(2), 1e-5);
            AssertNumber.AreEqual(Math.Exp(5.0), result.Get(3), 1e-5);
        }


        [TestMethod]
        public void Transpose()
        {
            var volume = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(1, 2, 1, 2));
            var result = NewVolume(new double[4], new Shape(2, 1, 1, 2));

            volume.Transpose(result);
            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(2.0, result.Get(1, 0, 0, 0));
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(4.0, result.Get(1, 0, 0, 1));
        }

        [TestMethod]
        public void Sqrt()
        {
            var volume = NewVolume(new[] { 0, 1, 3.0 * 3.0, 5.0 * 5.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            volume.Sqrt(result);
            AssertNumber.AreEqual(Math.Sqrt(0.0), result.Get(0), 1e-5);
            AssertNumber.AreEqual(Math.Sqrt(1.0), result.Get(1), 1e-5);
            AssertNumber.AreEqual(Math.Sqrt(9.0), result.Get(2), 1e-5);
            AssertNumber.AreEqual(Math.Sqrt(25.0), result.Get(3), 1e-5);
        }

        /// <summary>
        /// Make sure we can use input volume as output
        /// </summary>
        [TestMethod]
        public void SqrtInPlace()
        {
            var volume = NewVolume(new[] { 0, 1, 3.0 * 3.0, 5.0 * 5.0 }, new Shape(4));
            volume.Sqrt(volume);

            AssertNumber.AreEqual(Math.Sqrt(0.0), volume.Get(0), 1e-5);
            AssertNumber.AreEqual(Math.Sqrt(1.0), volume.Get(1), 1e-5);
            AssertNumber.AreEqual(Math.Sqrt(9.0), volume.Get(2), 1e-5);
            AssertNumber.AreEqual(Math.Sqrt(25.0), volume.Get(3), 1e-5);
        }

        [TestMethod]
        public void Tanh()
        {
            var volume = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            volume.Tanh(result);
            AssertNumber.AreEqual(Math.Tanh(-1.0), result.Get(0), 1e-6);
            AssertNumber.AreEqual(Math.Tanh(0.0), result.Get(1), 1e-6);
            AssertNumber.AreEqual(Math.Tanh(3.0), result.Get(2), 1e-6);
            AssertNumber.AreEqual(Math.Tanh(5.0), result.Get(3), 1e-6);
        }

        [TestMethod]
        public void TanhGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            inputActivation.Relu(outputActivation);
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));
            var result = NewVolume(new double[4], new Shape(4));

            outputActivation.TanhGradient(inputActivation, outputActivationGradient, result);

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

        /// <summary>
        /// Dropout should let the volume go thru if drop probability is 0
        /// </summary>
        [TestMethod]
        public void DropoutWith0Dropprob()
        {
            var volume = NewVolume(RandomUtilities.RandomDoubleArray(100), new Shape(100));
            var result = NewVolume(new double[100], new Shape(100));
            var dropprob = (T)Convert.ChangeType(0.0, typeof(T));

            // Forward
            volume.Dropout(dropprob, result);
            Assert.IsTrue(volume.ToArray().SequenceEqual(result.ToArray()));

            // Backward
            var inputGradient = BuilderInstance<T>.Volume.SameAs(volume.Storage, volume.Shape);
            var outputActivationGradient = NewVolume(new double[100].Populate(1.0), new Shape(100));
            result.DropoutGradient(volume, outputActivationGradient, inputGradient, dropprob);

            Assert.IsTrue(inputGradient.ToArray().SequenceEqual(outputActivationGradient.ToArray()));
        }

        [TestMethod]
        public void Dropout()
        {
            var volume = NewVolume(new double[100].Populate(1.0), new Shape(100));
            var result = NewVolume(new double[100], new Shape(100));

            var dropProb = 0.0;
            volume.Dropout((T)Convert.ChangeType(dropProb, typeof(T)), result);

            var array = result.Storage.ToArray();
            var c = array.Count(o => o.Equals(Ops<T>.Zero));
            Assert.IsTrue(dropProb > 0 ? c > 0 : c >= 0);

            // Check magnitude scale up
            var nonZeroEntry = array.First(o => !o.Equals(Ops<T>.Zero));
            AssertNumber.AreEqual(1.0 / (1 - dropProb), nonZeroEntry, 1e-6);

            var inputGradient = BuilderInstance<T>.Volume.SameAs(volume.Storage, volume.Shape);
            var gradient = 1.0;
            var outputActivationGradient = NewVolume(new double[100].Populate(gradient), new Shape(100));
            result.DropoutGradient(volume, outputActivationGradient, inputGradient, (T)Convert.ChangeType(dropProb, typeof(T)));

            array = inputGradient.Storage.ToArray();
            nonZeroEntry = array.First(o => !o.Equals(Ops<T>.Zero));
            AssertNumber.AreEqual(gradient / (1 - dropProb), nonZeroEntry, 1e-6);
        }
    }
}