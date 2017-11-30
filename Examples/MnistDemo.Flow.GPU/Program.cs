using System;
using System.IO;
using System.Linq;
using System.Windows;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Fluent;
using ConvNetSharp.Flow.Layers;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Utils.GraphVisualizer;
using ConvNetSharp.Volume;


namespace MnistDemo.GPU
{
    internal class Program
    {
        private readonly CircularBuffer<double> _testAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> _trainAccWindow = new CircularBuffer<double>(100);
        private Net<float> _net;
        private int _stepCount;
        private SgdTrainer<float> _trainer;

        [STAThread]
        private static void Main()
        {
            var program = new Program();
            program.MnistDemo();
        }

        private void MnistDemo()
        {
            BuilderInstance<float>.Volume = new ConvNetSharp.Volume.GPU.Single.VolumeBuilder();

            var datasets = new DataSets();
            if (!datasets.Load(100))
            {
                return;
            }

            // Create network
            this._net = new Net<float>();
            this._net.AddLayer(new InputLayer<float>());
            this._net.AddLayer(new ConvLayer<float>(5, 5, 8) { Stride = 1, Pad = 2, BiasPref = 0.1f });
            this._net.AddLayer(new ReluLayer<float>());
            this._net.AddLayer(new PoolLayer<float>(2, 2) { Stride = 2 });
            this._net.AddLayer(new ConvLayer<float>(5, 5, 16) { Stride = 1, Pad = 2, BiasPref = 0.1f });
            this._net.AddLayer(new ReluLayer<float>());
            this._net.AddLayer(new PoolLayer<float>(3, 3) { Stride = 3 });
            this._net.AddLayer(new FullyConnLayer<float>(10));
            this._net.AddLayer(new SoftmaxLayer<float>());

            // Fluent version
            //this._net = Net<float>.Create()
            //           .Conv(5, 5, 8).Stride(1).Pad(2)
            //           .Relu()
            //           .Pool(2, 2).Stride(2)
            //           .Conv(5, 5, 16).Stride(1).Pad(2)
            //           .Relu()
            //           .Pool(3, 3).Stride(3)
            //           .FullyConn(10)
            //           .Softmax()
            //           .Build();

            this._trainer = new SgdTrainer<float>(this._net, 0.01f)
            {
                BatchSize = 1024,
                //L2Decay = 0.001f,
                //Momentum = 0.9f
            };

            if (File.Exists("loss.csv"))
            {
                File.Delete("loss.csv");
            }

            Console.WriteLine("Convolutional neural network learning...[Press any key to stop]");
            do
            {
                var trainSample = datasets.Train.NextBatch(this._trainer.BatchSize);
                Train(trainSample.Item1, trainSample.Item2, trainSample.Item3);

                var testSample = datasets.Test.NextBatch(this._trainer.BatchSize);
                Test(testSample.Item1, testSample.Item3, this._testAccWindow);

                Console.WriteLine("Loss: {0} Train accuracy: {1}% Test accuracy: {2}%", this._trainer.Loss,
                    Math.Round(this._trainAccWindow.Items.Average() * 100.0, 2),
                    Math.Round(this._testAccWindow.Items.Average() * 100.0, 2));

                Console.WriteLine("Example seen: {0} Fwd: {1}ms Bckw: {2}ms Updt: {3}ms", this._stepCount,
                    Math.Round(this._trainer.ForwardTimeMs, 2),
                    Math.Round(this._trainer.BackwardTimeMs, 2),
                    Math.Round(this._trainer.UpdateWeightsTimeMs, 2));

                File.AppendAllLines("loss.csv", new[] { $"{this._stepCount}, {this._trainer.Loss}, { Math.Round(this._trainAccWindow.Items.Average() * 100.0, 2)}, {Math.Round(this._testAccWindow.Items.Average() * 100.0, 2)}" });

            } while (!Console.KeyAvailable);


            // Display graph
            var vm = new ViewModel<float>(_net.Op);
            var app = new Application();
            app.Run(new GraphControl { DataContext = vm });

            this._net.Dispose();
            this._trainer.Dispose();
        }

        private void Test(Volume<float> x, int[] labels, CircularBuffer<double> accuracy, bool forward = true)
        {
            if (forward)
            {
                this._net.Forward(x);
            }

            var prediction = this._net.GetPrediction();

            for (var i = 0; i < labels.Length; i++)
            {
                accuracy.Add(labels[i] == prediction[i] ? 1.0 : 0.0);
            }

            x.Dispose();
        }

        private void Train(Volume<float> x, Volume<float> y, int[] labels)
        {
            this._trainer.Train(x, y);

            Test(x, labels, this._trainAccWindow, false);

            this._stepCount += labels.Length;

            x.Dispose();
        }
    }
}