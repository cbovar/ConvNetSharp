using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp;

namespace MnistDemo
{
    internal class Program
    {
        private const int BatchSize = 3000;
        private readonly Random random = new Random();
        private readonly CircularBuffer<double> trainAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> valAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> wLossWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> xLossWindow = new CircularBuffer<double>(100);
        private Net net;
        private int stepCount;
        private List<MnistEntry> testing;
        private Trainer trainer;
        private List<MnistEntry> training;
        private int trainingCount = BatchSize;

        private void MnistDemo()
        {
            // Load data
            this.training = MnistReader.Load(@"..\..\Mnist\train-labels.idx1-ubyte", @"..\..\Mnist\train-images.idx3-ubyte");
            this.testing = MnistReader.Load(@"..\..\Mnist\t10k-labels.idx1-ubyte", @"..\..\Mnist\t10k-images.idx3-ubyte");

            if (this.training.Count == 0 || this.testing.Count == 0)
            {
                Console.WriteLine("Missing Mnist training/testing files.");
                Console.ReadKey();
                return;
            }

            // Create network
            this.net = new Net();
            this.net.AddLayer(new InputLayer(24, 24, 1));
            this.net.AddLayer(new ConvLayer(5, 5, 8) { Stride = 1, Pad = 2, Activation = Activation.Relu });
            this.net.AddLayer(new PoolLayer(2, 2) { Stride = 2 });
            this.net.AddLayer(new ConvLayer(5, 5, 16) { Stride = 1, Pad = 2, Activation = Activation.Relu });
            this.net.AddLayer(new PoolLayer(3, 3) { Stride = 3 });
            this.net.AddLayer(new SoftmaxLayer(10));

            this.trainer = new Trainer(this.net)
            {
                BatchSize = 20,
                L2Decay = 0.001,
                TrainingMethod = Trainer.Method.Adadelta
            };

            do
            {
                var sample = this.SampleTrainingInstance();
                this.Step(sample);
            } while (!Console.KeyAvailable);
        }

        private void Step(Item sample)
        {
            var x = sample.Volume;
            var y = sample.Label;

            if (sample.IsValidation)
            {
                // use x to build our estimate of validation error
                this.net.Forward(x);
                var yhat = this.net.GetPrediction();
                var valAcc = yhat == y ? 1.0 : 0.0;
                this.valAccWindow.Add(valAcc);
                return;
            }

            // train on it with network
            this.trainer.Train(x, y);
            var lossx = this.trainer.CostLoss;
            var lossw = this.trainer.L2DecayLoss;

            // keep track of stats such as the average training error and loss
            var prediction = this.net.GetPrediction();
            var trainAcc = prediction == y ? 1.0 : 0.0;
            this.xLossWindow.Add(lossx);
            this.wLossWindow.Add(lossw);
            this.trainAccWindow.Add(trainAcc);

            if (this.stepCount % 200 == 0)
            {
                if (this.xLossWindow.Count == this.xLossWindow.Capacity)
                {
                    var xa = this.xLossWindow.Items.Average();
                    var xw = this.wLossWindow.Items.Average();
                    var loss = xa + xw;

                    Console.WriteLine("Loss: {0} Train accuray: {1} Test accuracy: {2}", loss,
                        Math.Round(this.trainAccWindow.Items.Average() * 100.0, 2),
                        Math.Round(this.valAccWindow.Items.Average() * 100.0, 2));

                    Console.WriteLine("Example seen: {0} Fwd: {1}ms Bckw: {2}ms", this.stepCount,
                        Math.Round(this.trainer.ForwardTime.TotalMilliseconds, 2),
                        Math.Round(this.trainer.BackwardTime.TotalMilliseconds, 2));
                }
            }

            if (this.stepCount % 1000 == 0)
            {
                this.TestPredict();
            }

            this.stepCount++;
        }

        private void TestPredict()
        {
            for (var i = 0; i < 50; i++)
            {
                List<Item> sample = this.SampleTestingInstance();
                var y = sample[0].Label; // ground truth label

                // forward prop it through the network
                var average = new Volume(1, 1, 10, 0.0);
                var n = sample.Count;
                for (var j = 0; j < n; j++)
                {
                    var a = this.net.Forward(sample[j].Volume);
                    average.AddFrom(a);
                }

                var predictions = average.Weights.Select((w, k) => new { Label = k, Weight = w }).OrderBy(o => -o.Weight);
            }
        }

        private Item SampleTrainingInstance()
        {
            var n = this.random.Next(this.trainingCount);
            var entry = this.training[n];

            // load more batches over time
            if (this.stepCount % 5000 == 0 && this.stepCount > 0)
            {
                this.trainingCount = Math.Min(this.trainingCount + BatchSize, this.training.Count);
            }

            // Create volume from image data
            var x = new Volume(28, 28, 1, 0.0);

            for (var i = 0; i < 28; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    x.Weights[j + i * 28] = entry.Image[j + i * 28] / 255.0;
                }
            }

            x = x.Augment(24);

            return new Item { Volume = x, Label = entry.Label, IsValidation = n % 10 == 0 };
        }

        private List<Item> SampleTestingInstance()
        {
            var result = new List<Item>();
            var n = this.random.Next(this.testing.Count);
            var entry = this.testing[n];

            // Create volume from image data
            var x = new Volume(28, 28, 1, 0.0);

            for (var i = 0; i < 28; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    x.Weights[j + i * 28] = entry.Image[j + i * 28] / 255.0;
                }
            }

            for (var i = 0; i < 4; i++)
            {
                result.Add(new Item { Volume = x.Augment(24), Label = entry.Label });
            }

            return result;
        }

        private static void Main(string[] args)
        {
            var program = new Program();
            program.MnistDemo();
        }

        private class Item
        {
            public Volume Volume { get; set; }

            public int Label { get; set; }

            public bool IsValidation { get; set; }
        }
    }
}