using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.GPU.Double;

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
            var gpuVolumeBuilder = new VolumeBuilder();
            var cpuVolumeBuilder = new Volume.Double.VolumeBuilder();

            const int nmLayers = 3;
            const int layerSize = 30;
            const int nmSets = 12900;
            const int nmIterations = 1;
            var input = Shape.From(24, 1, 1);
            var output = 2;

            for (var batchSize = 10; batchSize < nmSets; batchSize *= 2)
            {
                Console.WriteLine($"-- {nameof(batchSize)} == {batchSize} ------------------");

                BuilderInstance<double>.Volume = cpuVolumeBuilder;
                var testNet = Create(layerSize, nmLayers, input, output);
                ExecuteNeuralNet("CPU", testNet, batchSize, nmSets, nmIterations);

                BuilderInstance<double>.Volume = gpuVolumeBuilder;
                testNet = Create(layerSize, nmLayers, input, output);
                ExecuteNeuralNet("GPU", testNet, batchSize, nmSets, nmIterations);

                Console.WriteLine();
            }
        }

        private static TestNet Create(int layerSize, int nmLayers, Shape input, int output)
        {
            var net = new TestNet { InputShape = new[] { Shape.From(input) }, OutputShape = Shape.From(1, 1, output) };

            net.AddLayer(new InputLayer(input.Dimensions[0], input.Dimensions[1], input.Dimensions[2]));
            for (var i = 0; i < nmLayers; i++)
            {
                net.AddLayer(new FullyConnLayer(layerSize));
                net.AddLayer(new ReluLayer());
            }

            net.AddLayer(new FullyConnLayer(output));
            net.AddLayer(new SoftmaxLayer(output));
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
                        var inputBatch = Shape.From(inputShape.Dimensions[0], inputShape.Dimensions[1], inputShape.Dimensions[2], batchSize);
                        return builder.Random(inputBatch);
                    }).ToArray();

                var outputShape = Shape.From(consumer.OutputShape.Dimensions[0], consumer.OutputShape.Dimensions[1], consumer.OutputShape.Dimensions[2], batchSize);
                var tempBatchOutputs = builder.Random(outputShape);
                var batchOutputs = builder.SameAs(outputShape);
                tempBatchOutputs.Softmax(batchOutputs);

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

            var trainer = new SgdTrainer(net) { LearningRate = 0.01, Momentum = 0.5, BatchSize = batchSize };

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