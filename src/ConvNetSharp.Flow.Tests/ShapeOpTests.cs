using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using ConvNetSharp.Core;
using ConvNetSharp.Flow.Layers;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using ConvNetSharp.Volume.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public class DoubleOpTests : OpTests<double>
    {
        public DoubleOpTests()
        {
            BuilderInstance<double>.Volume = new VolumeBuilder();
        }

        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new Volume.Double.Volume(values, shape);
        }
    }

    [TestClass]
    public class SingleOpTests : OpTests<float>
    {
        public SingleOpTests()
        {
            BuilderInstance<float>.Volume = new Volume.Single.VolumeBuilder();
        }

        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float)i).ToArray();
            return new Volume.Single.Volume(converted, shape);
        }
    }

    [TestClass]
    public class SingleGpuOpTests : OpTests<float>
    {
        public SingleGpuOpTests()
        {
            BuilderInstance<float>.Volume = new Volume.GPU.Single.VolumeBuilder();
        }

        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float)i).ToArray();
            return new Volume.GPU.Single.Volume(converted, shape);
        }
    }

    [TestClass]
    public class DoubleGpuOpTests : OpTests<double>
    {
        public DoubleGpuOpTests()
        {
            BuilderInstance<double>.Volume = new Volume.GPU.Double.VolumeBuilder();
        }

        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new Volume.GPU.Double.Volume(values, shape);
        }
    }
 
    [TestClass]
    public abstract class OpTests<T> where T : struct, IEquatable<T>, IFormattable
    {
        [TestMethod]
        public void CompareCoreVsFlow()
        {
            var inputWidth = 28;
            var inputHeigth = 28;
            var inputDepth = 3;
            var batchSize = 20;

            #region Flow network

            var netFlow = new Net<T>();
            netFlow.AddLayer(new InputLayer<T>());
            var convLayerFlow1 = new ConvLayer<T>(5, 5, 8) { BiasPref = (T)Convert.ChangeType(0.1, typeof(T)), Stride = 1, Pad = 2 };
            netFlow.AddLayer(convLayerFlow1);
            netFlow.AddLayer(new ReluLayer<T>());
            netFlow.AddLayer(new PoolLayer<T>(2, 2) { Stride = 2 });
            var fullyConnLayerFlow = new FullyConnLayer<T>(10);
            netFlow.AddLayer(fullyConnLayerFlow);
            netFlow.AddLayer(new SoftmaxLayer<T>());

            var trainerFlow = new SgdTrainer<T>(netFlow, (T)Convert.ChangeType(0.01f, typeof(T)))
            {
                BatchSize = batchSize
            };

            #endregion

            #region Core network

            var netCore = new Core.Net<T>();
            netCore.AddLayer(new Core.Layers.InputLayer<T>(inputWidth, inputHeigth, inputDepth));
            var convLayerCore1 = new Core.Layers.ConvLayer<T>(5, 5, 8) { BiasPref = (T)Convert.ChangeType(0.1, typeof(T)), Stride = 1, Pad = 2 };
            netCore.AddLayer(convLayerCore1);
            netCore.AddLayer(new Core.Layers.ReluLayer<T>());
            netCore.AddLayer(new Core.Layers.PoolLayer<T>(2, 2) { Stride = 2 });
            var fullyConnLayerCore = new Core.Layers.FullyConnLayer<T>(10);
            netCore.AddLayer(fullyConnLayerCore);
            netCore.AddLayer(new Core.Layers.SoftmaxLayer<T>(10));

            var trainerCore = new Core.Training.SgdTrainer<T>(netCore)
            {
                LearningRate = (T)Convert.ChangeType(0.01f, typeof(T)),
                BatchSize = batchSize
            };

            #endregion

            // Same weights
            var convfilterCore1 = netFlow.Session.GetVariableByName(netFlow.Op, "ConvLayer_1/Filter_1");
            convfilterCore1.Result = BuilderInstance<T>.Volume.SameAs(convLayerCore1.Filters.ToArray(), convLayerCore1.Filters.Shape);

            var fullyfilterCore = netFlow.Session.GetVariableByName(netFlow.Op, "FullConnLayer_4/Filter_2");
            fullyfilterCore.Result = BuilderInstance<T>.Volume.SameAs(fullyConnLayerCore.Filters.ToArray(), fullyConnLayerCore.Filters.Shape);

            // Create input
            var xStorage = new double[inputWidth * inputHeigth * inputDepth * batchSize].Populate(1.0);
            var x = NewVolume(xStorage, Volume.Shape.From(inputWidth, inputHeigth, inputDepth, batchSize));

            // Create output
            var yStorage = new double[10 * batchSize];
            var y = NewVolume(yStorage, Volume.Shape.From(1, 1, 10, batchSize));
            for (var i = 0; i < batchSize; i++)
            {
                y.Set(0, 0, i % 10, i, Ops<T>.One);
            }

            for (var k = 0; k < 10; k++)
            {
                xStorage = new double[inputWidth * inputHeigth * inputDepth * batchSize].Populate(1.0 + k);
                x = NewVolume(xStorage, Volume.Shape.From(inputWidth, inputHeigth, inputDepth, batchSize));

                var flowResult = netFlow.Forward(x);
                var coreResult = netCore.Forward(x);

                var sum1 = BuilderInstance<T>.Volume.SameAs(new Shape(1));
                flowResult.DoSum(sum1);
                var sum2 = BuilderInstance<T>.Volume.SameAs(new Shape(1));
                coreResult.DoSum(sum2);
                var diff = Ops<T>.Subtract(sum1.Get(0), sum2.Get(0));

                Console.WriteLine(diff);

                AssertNumber.AreSequenceEqual(flowResult.ToArray(), coreResult.ToArray(), 1e-6);

                trainerCore.Train(x, y);
                trainerFlow.Train(x, y);
            }
        }

        protected abstract Volume<T> NewVolume(double[] values, Shape shape);

        [TestMethod]
        public void Reshape()
        {
            var x = new PlaceHolder<T>("x");
            var op = new Reshape<T>(x, new Shape(1, 1, -1, 1));

            using (var session = new Session<T>())
            {
                // [4] -> [1,1,4,1]
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(4)) } });
                Assert.AreEqual(new Shape(1, 1, 4, 1), result.Shape);

                // [8] -> [1,1,8,1]
                result = session.Run(op, new Dictionary<string, Volume<T>>
                {
                    {
                        "x", NewVolume(new[]
                        {
                            1.0, 2.0, 3.0, 4.0,
                            1.0, 2.0, 3.0, 4.0
                        }, Volume.Shape.From(8))
                    }
                });
                Assert.AreEqual(new Shape(1, 1, 8, 1), result.Shape);
            }
        }

        [TestMethod]
        public void ReshapeDerivate()
        {
            var x = new PlaceHolder<T>("x");
            var op = new Reshape<T>(x, new Shape(1, 1, -1, 1));
            var grad = new PlaceHolder<T>("grad");

            using (var session = new Session<T>())
            {
                op.Derivate = grad;
                op.Differentiate();

                var diff = x.Derivate;

                // [4,1,1,1] -> [1,1,4,1]
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(4, 1, 1, 1)) } });

                // [1,1,4,1] -> [4,1,1,1]
                result = session.Run(diff,
                    new Dictionary<string, Volume<T>>
                    {
                        {"x", NewVolume(new[] {1.0, 2.0, 3.0, 4.0}, Volume.Shape.From(4))},
                        {"grad", NewVolume(new[] {1.0, 1.0, 1.0, 1.0}, Volume.Shape.From(1, 1, 4, 1))}
                    });
                Assert.AreEqual(new Shape(4, 1, 1, 1), result.Shape);
            }
        }

        [TestMethod]
        public void Shape()
        {
            var x = new PlaceHolder<T>("x");
            var op = new Shape<T>(x);

            using (var session = new Session<T>())
            {
                // Batch size = 1
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(1, 1, 4, 1)) } });

                AssertNumber.AreEqual(1.0, result.Get(0));
                AssertNumber.AreEqual(1.0, result.Get(1));
                AssertNumber.AreEqual(4.0, result.Get(2));
                AssertNumber.AreEqual(1.0, result.Get(3));

                // Batch size = 2
                result = session.Run(op, new Dictionary<string, Volume<T>>
                {
                    {
                        "x", NewVolume(new[]
                        {
                            1.0, 2.0, 3.0, 4.0,
                            1.0, 2.0, 3.0, 4.0
                        }, Volume.Shape.From(1, 1, 4, 2))
                    }
                });

                AssertNumber.AreEqual(1.0, result.Get(0));
                AssertNumber.AreEqual(1.0, result.Get(1));
                AssertNumber.AreEqual(4.0, result.Get(2));
                AssertNumber.AreEqual(2.0, result.Get(3));
            }
        }
    }
}