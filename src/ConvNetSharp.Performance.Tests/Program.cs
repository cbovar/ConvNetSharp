using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Performance.Tests
{
    public class TestNet : Net<double>
    {
        public Shape[] InputShape { get; set; }
        public Shape OutputShape { get; set; }
    }

    public class Set
    {
        public Volume<double>[] Inputs { get; set; }
        public Volume<double> Outputs { get; set; }
    }

    public static class Program
    {
        public static void Main(string[] args)
        {
            var gpuVolumeBuilder = new Volume.GPU.Double.VolumeBuilder();
            var cpuVolumeBuilder = new Volume.Double.VolumeBuilder();

            BuilderInstance<double>.Volume = cpuVolumeBuilder;
            var testNet = Create(20, 4, 4);
            ExecuteNeuralNet("CPU", testNet, 100, 1000, 10);

            BuilderInstance<double>.Volume = gpuVolumeBuilder;
            testNet = Create(20, 4, 4);
            ExecuteNeuralNet("GPU", testNet, 100, 1000, 10);
        }

        private static TestNet Create(int layerSize, int nmLayers, int inputWHD)
        {
            var net = new TestNet();
            net.InputShape = new[] { Shape.From(inputWHD, inputWHD, inputWHD) };
            net.OutputShape = Shape.From(1, 1, layerSize);
            net.AddLayer(new InputLayer(inputWHD, inputWHD, inputWHD));
            for (var i = 0; i < nmLayers; i++)
            {
                net.AddLayer(new FullyConnLayer(layerSize));
                net.AddLayer(new SigmoidLayer());
            }
            net.AddLayer(new FullyConnLayer(layerSize));
            net.AddLayer(new SoftmaxLayer(layerSize));
            return net;
        }

        public static Set[] CreateSampleSets(
            TestNet consumer,
            int batchSize,
            int totalSets)
        {
            var sets = new List<Set>();

            var builder = BuilderInstance<double>.Volume;

            for (var s = 0; s < totalSets; s += batchSize)
            {
                var batchInputs = consumer
                    .InputShape
                    .Select(inputShape =>
                    {
                        var inputBatch = Shape.From(inputShape, batchSize);
                        return builder.Random(inputBatch);
                    }).ToArray();

                var outputShape = Shape.From(consumer.OutputShape, batchSize);
                var tempBatchOutputs = builder.Random(outputShape);
                var batchOutputs = builder.SameAs(outputShape);
                tempBatchOutputs.DoSoftMax(batchOutputs);

                sets.Add(new Set
                {
                    Inputs = batchInputs,
                    Outputs = batchOutputs
                });
            }

            return sets.ToArray();
        }

        private static void ExecuteNeuralNet(
            string name,
            TestNet net,
            int batchSize,
            int totalSets,
            int iterations)
        {
            var inputs = CreateSampleSets(net, batchSize, totalSets);

            var stopWatch = new Stopwatch();
            Console.WriteLine($"- {name} ------");
            stopWatch.Restart();

            var trainer = new SgdTrainer(net);
            trainer.LearningRate = 0.01;
            trainer.Momentum = 0;
            trainer.L1Decay = 0;
            trainer.L2Decay = 0;
            trainer.BatchSize = batchSize;

            for (var i = 0; i < iterations; i++)
            {
                foreach (var set in inputs)
                {
                    trainer.Train(set.Inputs[0], set.Outputs);
                }
            }

            stopWatch.Stop();

            Console.WriteLine("    total: {0:0.000}ms", stopWatch.ElapsedMilliseconds);
            Console.WriteLine("  forward: {0:0.000}ms", trainer.ForwardTimeMs);
            Console.WriteLine(" backward: {0:0.000}ms", trainer.BackwardTimeMs);
            Console.WriteLine("   update: {0:0.000}ms", trainer.UpdateWeightsTimeMs);
        }
    }
}
