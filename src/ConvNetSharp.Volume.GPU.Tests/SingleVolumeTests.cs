using System;
using System.Linq;
using ConvNetSharp.Volume.GPU.Single;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestClass]
    public class SingleVolumeTests
    {
        [TestMethod]
        public void Add1D()
        {
            var left = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f }, new Shape(3),
                GpuContext.Default);
            var right = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f }, new Shape(3),
                GpuContext.Default);

            var result = left + right;
            Assert.AreEqual(2.0f, result.Get(0));
            Assert.AreEqual(4.0f, result.Get(1));
            Assert.AreEqual(6.0f, result.Get(2));
        }

        [TestMethod]
        public void Add2D()
        {
            var left = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new Shape(2, -1),
                GpuContext.Default);
            var right = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new Shape(2, -1),
                GpuContext.Default);

            var result = left + right;
            Assert.AreEqual(2.0f, result.Get(0, 0));
            Assert.AreEqual(4.0f, result.Get(1, 0));
            Assert.AreEqual(6.0f, result.Get(0, 1));
            Assert.AreEqual(8.0f, result.Get(1, 1));
        }

        [TestMethod]
        public void AddBroadcast()
        {
            var volume = new Single.Volume(new[]
            {
                1.0f, 2.0f,
                3.0f, 4.0f,
                1.0f, 2.0f,
                3.0f, 4.0f,
                1.0f, 2.0f,
                3.0f, 4.0f
            }, new Shape(2, 2, 3), GpuContext.Default);

            var bias = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f }, new Shape(1, 1, 3), GpuContext.Default);

            var result = volume + bias;
            Assert.AreEqual(2.0f, result.Get(0, 0, 0));
            Assert.AreEqual(3.0f, result.Get(0, 0, 1));
            Assert.AreEqual(4.0f, result.Get(0, 0, 2));
        }

        [TestMethod]
        public void BiasBackward()
        {
            var outputGradient = new Single.Volume(
                new[]
                {
                    1.0f, 2.0f,
                    3.0f, 1.0f,
                    2.0f, 3.0f
                },
                new Shape(2, 1, 3, 1),
                GpuContext.Default);

            var biasGradient = BuilderInstance<float>.Volume.SameAs(new Shape(1, 1, 3, 1));

            outputGradient.BiasGradient(biasGradient);

            Assert.AreEqual(3.0f, biasGradient.Get(0, 0, 0, 0));
            Assert.AreEqual(4.0f, biasGradient.Get(0, 0, 1, 0));
            Assert.AreEqual(5.0f, biasGradient.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void BiasBackwardBatch()
        {
            var outputGradient = new Single.Volume(
                new[]
                {
                    1.0f, 2.0f,
                    3.0f, 1.0f,
                    2.0f, 3.0f,
                    1.0f, 2.0f,
                    3.0f, 1.0f,
                    2.0f, 3.0f
                },
                new Shape(2, 1, 3, 2),
                GpuContext.Default);

            var biasGradient = BuilderInstance<float>.Volume.SameAs(new Shape(1, 1, 3, 1));

            outputGradient.BiasGradient(biasGradient);

            Assert.AreEqual(6,0f, biasGradient.Get(0, 0, 0, 0));
            Assert.AreEqual(8.0f, biasGradient.Get(0, 0, 1, 0));
            Assert.AreEqual(10.0f, biasGradient.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void Builder()
        {
            var example = new Single.Volume(new[] { 1.0f }, new Shape(1), GpuContext.Default);
            var volume = BuilderInstance<float>.Volume.SameAs(example.Storage, 1.0f, new Shape(10));

            // SameAs creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is filled with provided value
            Assert.AreEqual(10, volume.Shape.GetDimension(0));
            for (var i = 0; i < 10; i++)
            {
                Assert.AreEqual(1.0f, volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderArray()
        {
            var array = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            var volume = BuilderInstance<float>.Volume.SameAs(array, new Shape(5));

            Assert.AreEqual(5, volume.Shape.GetDimension(0));
            for (var i = 0; i < 5; i++)
            {
                Assert.AreEqual(array[i], volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderEmpty()
        {
            var example = new Single.Volume(new[] { 1.0f }, new Shape(1), GpuContext.Default);
            var volume = BuilderInstance<float>.Volume.SameAs(example.Storage, new Shape(10));

            // SameAs creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is empty
            Assert.AreEqual(10, volume.Shape.GetDimension(0));
            for (var i = 0; i < 10; i++)
            {
                Assert.AreEqual(0.0f, volume.Get(i));
            }
        }

        [ClassInitialize]
        public static void ClassInit(TestContext context)
        {
            BuilderInstance<float>.Volume = new VolumeBuilder();
        }

        [TestMethod]
        public void ConvolveGradient()
        {
            // 3x3x3x1
            var input = new Single.Volume(new float[27].Populate(1.0f), new Shape(3, 3, 3, 1),
                GpuContext.Default);

            // 2x2x3x2
            var filter = new Single.Volume(
                new float[12].Populate(1.0f).Concat(new float[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2), GpuContext.Default);

            var outputGradient = new Single.Volume(new[] { 2.0f, 3.0f }, new Shape(1, 1, 2, 1),
                GpuContext.Default);

            var inputGradient = BuilderInstance<float>.Volume.SameAs(input.Storage, input.Shape);
            var filterGradient = BuilderInstance<float>.Volume.SameAs(filter.Storage, filter.Shape);

            input.ConvolveGradient(filter, outputGradient, inputGradient, filterGradient, 0, 2);

            Assert.AreEqual(8, inputGradient.Get(0, 0, 0, 0));
            Assert.AreEqual(0, inputGradient.Get(2, 2, 2, 0));
            Assert.AreEqual(0, inputGradient.Get(2, 2, 1, 0));
        }

        [TestMethod]
        public void Convolve()
        {
            // 3x3x3x1
            var input = new Single.Volume(new float[27].Populate(1.0f), new Shape(3, 3, 3, 1),
                GpuContext.Default);

            // 2x2x3x2
            var filter = new Single.Volume(
                new float[12].Populate(1.0f).Concat(new float[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2), GpuContext.Default);

            var result = input.Convolve(filter, 0, 2);

            // 1x1x2x1
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(1, result.Shape.GetDimension(3));

            Assert.AreEqual(12.0f, result.Storage.Get(0, 0, 0));
            Assert.AreEqual(24.0f, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void ConvolveBatch()
        {
            // 3x3x3x2
            var input = new Single.Volume(new float[27 * 2].Populate(1.0f), new Shape(3, 3, 3, 2),
                GpuContext.Default);

            // 2x2x3x2
            var filter = new Single.Volume(
                new float[12].Populate(1.0f).Concat(new float[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2),
                GpuContext.Default);

            var result = input.Convolve(filter, 0, 2);

            // 1x1x2x2
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(2, result.Shape.GetDimension(3));

            Assert.AreEqual(12.0f, result.Storage.Get(0, 0, 0, 0));
            Assert.AreEqual(24.0f, result.Storage.Get(0, 0, 1, 0));
            Assert.AreEqual(12.0f, result.Storage.Get(0, 0, 0, 1));
            Assert.AreEqual(24.0f, result.Storage.Get(0, 0, 1, 1));
        }

        [TestMethod]
        public void ConvolveGradientBatch()
        {
            // 3x3x3x2
            var input = new Single.Volume(new float[27 * 2].Populate(1.0f), new Shape(3, 3, 3, 2),
                GpuContext.Default);

            // 2x2x3x2
            var filter = new Single.Volume(
                new float[12].Populate(1.0f).Concat(new float[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2),
                GpuContext.Default);

            var outputGradient = new Single.Volume(new[]
                {
                    2.0f, 3.0f,
                    4.0f, 5.0f
                }, new Shape(1, 1, 2, 2),
                GpuContext.Default);

            var inputGradient = BuilderInstance<float>.Volume.SameAs(input.Storage, input.Shape);
            var filterGradient = BuilderInstance<float>.Volume.SameAs(filter.Storage, filter.Shape);

            input.ConvolveGradient(filter, outputGradient, inputGradient, filterGradient, 0, 2);

            // input gradient
            Assert.AreEqual(8.0f, inputGradient.Get(0, 0, 0, 0));
            Assert.AreEqual(0f, inputGradient.Get(2, 2, 2, 0));
            Assert.AreEqual(0f, inputGradient.Get(2, 2, 1, 0));
            Assert.AreEqual(14.0f, inputGradient.Get(0, 0, 0, 1));
            Assert.AreEqual(0.0f, inputGradient.Get(2, 2, 2, 1));
            Assert.AreEqual(0.0f, inputGradient.Get(2, 2, 1, 1));

            // filter gradient
            Assert.AreEqual(1.0f, filter.Get(0, 0, 0, 0));
            Assert.AreEqual(1.0f, filter.Get(0, 0, 1, 0));
            Assert.AreEqual(1.0f, filter.Get(0, 0, 2, 0));
            Assert.AreEqual(2.0f, filter.Get(0, 0, 0, 1));
            Assert.AreEqual(2.0f, filter.Get(0, 0, 1, 1));
            Assert.AreEqual(2.0f, filter.Get(0, 0, 2, 1));
        }

        /// <summary>
        ///     Fully connection can be expressed as a convolution with 1x1 filters
        /// </summary>
        [TestMethod]
        public void FullyCon()
        {
            // 1x3x1x1
            var input = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f }, new Shape(1, 1, 3, 1),
                GpuContext.Default);

            // 1x1x3x2
            var filter = new Single.Volume(
                new[] { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f },
                new Shape(1, 1, 3, 2), GpuContext.Default);

            var result = input.Convolve(filter, 0, 1);

            // 1x1x2x1
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(1, result.Shape.GetDimension(3));

            Assert.AreEqual(6.0f, result.Storage.Get(0, 0, 0));
            Assert.AreEqual(12.0f, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void Negate()
        {
            var volume = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f }, new Shape(3),
                GpuContext.Default);

            var result = -volume;
            Assert.AreEqual(-1.0f, result.Get(0));
            Assert.AreEqual(-2.0f, result.Get(1));
            Assert.AreEqual(-3.0f, result.Get(2));
        }

        [TestMethod]
        public void Pool2D()
        {
            var volume = new Single.Volume(new[]
            {
                1.0f, 0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 1.0f, 7.0f,
                2.0f, 0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 4.0f, 1.0f
            }, new Shape(4, 4), GpuContext.Default);

            var result = volume.Pool(2, 2, 0, 2);

            Assert.AreEqual(2, result.Shape.GetDimension(0));
            Assert.AreEqual(2, result.Shape.GetDimension(0));

            Assert.AreEqual(1.0f, result.Get(0, 0));
            Assert.AreEqual(7.0f, result.Get(1, 0));
            Assert.AreEqual(2.0f, result.Get(0, 1));
            Assert.AreEqual(4.0f, result.Get(1, 1));
        }

        [TestMethod]
        public void Pool2DBatch()
        {
            var volume = new Single.Volume(new[]
            {
                1.0f, 0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 1.0f, 7.0f,
                2.0f, 0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 4.0f, 1.0f,

                2.0f, 0.0f, 2.0f, 2.0f,
                2.0f, 0.0f, 2.0f, 14.0f,
                4.0f, 0.0f, 2.0f, 2.0f,
                2.0f, 0.0f, 8.0f, 2.0f
            }, new Shape(4, 4, 1, 2), GpuContext.Default);

            var result = volume.Pool(2, 2, 0, 2);

            Assert.AreEqual(2, result.Shape.GetDimension(0));
            Assert.AreEqual(2, result.Shape.GetDimension(1));
            Assert.AreEqual(1, result.Shape.GetDimension(2));
            Assert.AreEqual(2, result.Shape.GetDimension(3));

            Assert.AreEqual(1.0f, result.Get(0, 0, 0, 0));
            Assert.AreEqual(7.0f, result.Get(1, 0, 0, 0));
            Assert.AreEqual(2.0f, result.Get(0, 1, 0, 0));
            Assert.AreEqual(4.0f, result.Get(1, 1, 0, 0));

            Assert.AreEqual(2.0f, result.Get(0, 0, 0, 1));
            Assert.AreEqual(14.0f, result.Get(1, 0, 0, 1));
            Assert.AreEqual(4.0f, result.Get(0, 1, 0, 1));
            Assert.AreEqual(8.0f, result.Get(1, 1, 0, 1));
        }

        [TestMethod]
        public void Pool2DGradient()
        {
            var inputActivation = new Single.Volume(new[]
            {
                1.0f, 0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 1.0f, 7.0f,
                2.0f, 0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 4.0f, 1.0f
            }, new Shape(4, 4), GpuContext.Default);

            var outputActivation = inputActivation.Pool(2, 2, 0, 2);

            var outputActivationGradient = new Single.Volume(new[] { 1.0f, 1.0f, 1.0f, 1.0f }, new Shape(2, 2),
                GpuContext.Default);

            var result = outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 2);

            Assert.AreEqual(1.0f, result.Get(0, 0));
            Assert.AreEqual(1.0f, result.Get(3, 1));
            Assert.AreEqual(1.0f, result.Get(0, 2));
            Assert.AreEqual(1.0f, result.Get(2, 3));
        }

        [TestMethod]
        public void Pool2DGradientBatch()
        {
            var inputActivation = new Single.Volume(new[]
            {
                1.0f, 0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 1.0f, 7.0f,
                2.0f, 0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 4.0f, 1.0f,

                2.0f, 0.0f, 2.0f, 2.0f,
                2.0f, 0.0f, 2.0f, 14.0f,
                4.0f, 0.0f, 2.0f, 2.0f,
                2.0f, 0.0f, 8.0f, 2.0f
            }, new Shape(4, 4, 1, 2), GpuContext.Default);

            var outputActivation = inputActivation.Pool(2, 2, 0, 2);

            var outputActivationGradient = new Single.Volume(new[]
            {
                1.0f, 1.0f, 1.0f, 1.0f,
                2.0f, 2.0f, 2.0f, 2.0f,
            }, new Shape(2, 2, 1, 2), GpuContext.Default);

            var result = outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 2);

            Assert.AreEqual(1.0f, result.Get(0, 0, 0, 0));
            Assert.AreEqual(1.0f, result.Get(3, 1, 0, 0));
            Assert.AreEqual(1.0f, result.Get(0, 2, 0, 0));
            Assert.AreEqual(1.0f, result.Get(2, 3, 0, 0));

            Assert.AreEqual(2.0f, result.Get(0, 0, 0, 1));
            Assert.AreEqual(2.0f, result.Get(3, 1, 0, 1));
            Assert.AreEqual(2.0f, result.Get(0, 2, 0, 1));
            Assert.AreEqual(2.0f, result.Get(2, 3, 0, 1));
        }

        [TestMethod]
        public void Relu()
        {
            var volume = new Single.Volume(new[] { -1.0f, 0.0f, 3.0f, 5.0f }, new Shape(4), GpuContext.Default);

            var result = volume.Relu();
            Assert.AreEqual(0.0f, result.Get(0));
            Assert.AreEqual(0.0f, result.Get(1));
            Assert.AreEqual(3.0f, result.Get(2));
            Assert.AreEqual(5.0f, result.Get(3));
        }

        [TestMethod]
        public void ReluGradient()
        {
            var inputActivation = new Single.Volume(new[] { -1.0f, 0.0f, 3.0f, 5.0f }, new Shape(4),
                GpuContext.Default);
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = new Single.Volume(new[] { 1.0f, 1.0f, 1.0f, 1.0f }, new Shape(4),
                GpuContext.Default);

            var result = outputActivation.ReluGradient(inputActivation, outputActivationGradient);

            Assert.AreEqual(0.0f, result.Get(0));
            Assert.AreEqual(0.0f, result.Get(1));
            Assert.AreEqual(1.0f, result.Get(2));
            Assert.AreEqual(1.0f, result.Get(3));
        }

        [TestMethod]
        public void Shape2D()
        {
            var volume = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new Shape(2, -1),
                GpuContext.Default);
            Assert.AreEqual(2, volume.Shape.GetDimension(0));
            Assert.AreEqual(2, volume.Shape.GetDimension(1));
        }

        [TestMethod]
        public void Sigmoid()
        {
            var volume = new Single.Volume(new[] { -1.0f, 0.0f, 3.0f, 5.0f }, new Shape(4),
                GpuContext.Default);

            var result = volume.Sigmoid();
            Assert.AreEqual((float)(1.0 / (1.0 + Math.Exp(1.0))), result.Get(0));
            Assert.AreEqual((float)(1.0 / (1.0 + Math.Exp(0.0))), result.Get(1));
            Assert.AreEqual((float)(1.0 / (1.0 + Math.Exp(-3.0))), result.Get(2));
            Assert.AreEqual((float)(1.0 / (1.0 + Math.Exp(-5.0))), result.Get(3));
        }


        [TestMethod]
        public void SigmoidGradient()
        {
            var inputActivation = new Single.Volume(new[] { -1.0f, 0.0f, 3.0f, 5.0f }, new Shape(4), GpuContext.Default);
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = new Single.Volume(new[] { 1.0f, 1.0f, 1.0f, 1.0f }, new Shape(4),
                GpuContext.Default);

            var result = outputActivation.SigmoidGradient(inputActivation, outputActivationGradient);

            Assert.AreEqual(0.0f, result.Get(0));
            Assert.AreEqual(0.0f, result.Get(1));
            Assert.AreEqual(-6.0f, result.Get(2));
            Assert.AreEqual(-20.0f, result.Get(3));
        }

        [TestMethod]
        public void SoftMax()
        {
            var volume1 = new Single.Volume(new[] { 0.0f, 0.0f, 0.0f, 10000.0f }, new Shape(1, 1, -1, 1),
                GpuContext.Default);
            var softmax1 = volume1.SoftMax();
            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 0, 0));
            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 1, 0));
            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 2, 0));
            Assert.AreEqual(1.0f, softmax1.Get(0, 0, 3, 0));

            var volume2 = new Single.Volume(new[] { 10000.0f, 0.0f, 0.0f, 10000.0f }, new Shape(1, 1, -1, 1),
                GpuContext.Default);
            var softmax2 = volume2.SoftMax();
            Assert.AreEqual(0.5f, softmax2.Get(0, 0, 0, 0));
            Assert.AreEqual(0.5f, softmax2.Get(0, 0, 3, 0));
        }

        [TestMethod]
        public void SoftMaxBatch()
        {
            var input = new Single.Volume(new[]
            {
                0.0f, 0.0f, 0.0f, 10000.0f,
                0.0f, 0.0f, 10000.0f, 0.0f
            }, new Shape(1, 1, -1, 2), GpuContext.Default);
            var softmax1 = input.SoftMax();

            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 0, 0));
            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 1, 0));
            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 2, 0));
            Assert.AreEqual(1.0f, softmax1.Get(0, 0, 3, 0));

            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 0, 1));
            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 1, 1));
            Assert.AreEqual(1.0f, softmax1.Get(0, 0, 2, 1));
            Assert.AreEqual(0.0f, softmax1.Get(0, 0, 3, 1));
        }

        [TestMethod]
        public void SoftMaxGradient()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = new Single.Volume(new[] { 1.0f, 0.1f, 0.1f, 0.1f }, new Shape(1, 1, -1, 1),
                GpuContext.Default);

            // output  = softmax(input)
            var output = input.SoftMax();

            // groundTruth = [0, 1, 0 , 0]
            var groundTruth = new Single.Volume(new[] { 0.0f, 1.0f, 0.0f, 0.0f }, new Shape(1, 1, -1, 1),
                GpuContext.Default);

            // output gradient = 1 - groundTruth ./ output
            var outputGradient = new Single.Volume(new float[4], new Shape(1, 1, -1, 1), GpuContext.Default);
            groundTruth.Storage.Map((p, q) => 1 - p / q, output.Storage, outputGradient.Storage);

            // inputGradient = softmax_gradient(output, outputGradient)
            var inputGradient = output.SoftMaxGradient(outputGradient);

            // theorical result = output-groundTruth
            var result = output - groundTruth;

            Assert.AreEqual(result.Get(0, 0, 0, 0), inputGradient.Get(0, 0, 0, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 1, 0), inputGradient.Get(0, 0, 1, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 2, 0), inputGradient.Get(0, 0, 2, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 3, 0), inputGradient.Get(0, 0, 3, 0), 1e-4);
        }

        [TestMethod]
        public void SoftMaxGradientBatch()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = new Single.Volume(new[]
                {
                    1.0f, 0.1f, 0.1f, 0.1f,
                    0.1f, 0.1f, 1.0f, 0.1f
                }, new Shape(1, 1, -1, 2),
                GpuContext.Default);

            // output  = softmax(input)
            var output = input.SoftMax();

            // groundTruth = [0, 1, 0 , 0]
            var groundTruth = new Single.Volume(new[]
                {
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f
                }, new Shape(1, 1, -1, 2),
                GpuContext.Default);

            // output gradient = 1 - groundTruth ./ output
            var outputGradient = new Single.Volume(new float[8], new Shape(1, 1, -1, 2),
                GpuContext.Default);
            groundTruth.Storage.Map((p, q) => 1 - p / q, output.Storage, outputGradient.Storage);

            // inputGradient = softmax_gradient(output, outputGradient)
            var inputGradient = output.SoftMaxGradient(outputGradient);

            // theorical result = output-groundTruth
            var result = output - groundTruth;

            Assert.AreEqual(result.Get(0, 0, 0, 0), inputGradient.Get(0, 0, 0, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 1, 0), inputGradient.Get(0, 0, 1, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 2, 0), inputGradient.Get(0, 0, 2, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 3, 0), inputGradient.Get(0, 0, 3, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 0, 1), inputGradient.Get(0, 0, 0, 1), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 1, 1), inputGradient.Get(0, 0, 1, 1), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 2, 1), inputGradient.Get(0, 0, 2, 1), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 3, 1), inputGradient.Get(0, 0, 3, 1), 1e-4);
        }

        [TestMethod]
        public void SubstractFrom()
        {
            var left = new Single.Volume(new[] { 1.0f, 2.0f, 3.0f }, new Shape(3),
                GpuContext.Default);
            var right = new Single.Volume(new[] { 2.0f, 0.0f, 1.0f }, new Shape(3),
                GpuContext.Default);

            var result = left - right;
            Assert.AreEqual(-1.0f, result.Get(0));
            Assert.AreEqual(2.0f, result.Get(1));
            Assert.AreEqual(2.0f, result.Get(2));
        }

        [TestMethod]
        public void Tanh()
        {
            var volume = new Single.Volume(new[] { -1.0f, 0.0f, 3.0f, 5.0f }, new Shape(4),
                GpuContext.Default);

            var result = volume.Tanh();
            Assert.AreEqual((float)Math.Tanh(-1.0), result.Get(0));
            Assert.AreEqual((float)Math.Tanh(0.0), result.Get(1));
            Assert.AreEqual((float)Math.Tanh(3.0), result.Get(2));
            Assert.AreEqual((float)Math.Tanh(5.0), result.Get(3));
        }

        [TestMethod]
        public void TanhGradient()
        {
            var inputActivation = new Single.Volume(new[] { -1.0f, 0.0f, 3.0f, 5.0f }, new Shape(4), GpuContext.Default);
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = new Single.Volume(new[] { 1.0f, 1.0f, 1.0f, 1.0f }, new Shape(4),
                GpuContext.Default);

            var result = outputActivation.TanhGradient(inputActivation, outputActivationGradient);

            Assert.AreEqual(1.0f, result.Get(0));
            Assert.AreEqual(1.0f, result.Get(1));
            Assert.AreEqual(-8.0f, result.Get(2));
            Assert.AreEqual(-24.0f, result.Get(3));
        }

        [TestMethod]
        public void ToArray()
        {
            var floats = new[] { 1.0f, 2.0f, 3.0f };
            var v = new Single.Volume(floats, new Shape(3), GpuContext.Default);

            var array = v.ToArray();

            Assert.IsTrue(floats.SequenceEqual(array));
        }
    }
}