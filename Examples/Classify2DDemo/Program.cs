using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;

namespace Classify2DDemo
{
    internal class Program
    {
        private static void Classify2DDemo()
        {
            var net = new Net<double>();
            net.AddLayer(new InputLayer(1, 1, 2));
            net.AddLayer(new FullyConnLayer(6));
            net.AddLayer(new TanhLayer());
            net.AddLayer(new FullyConnLayer(2));
            net.AddLayer(new TanhLayer());
            net.AddLayer(new FullyConnLayer(2));
            net.AddLayer(new SoftmaxLayer(2));

            // Data
            var data = new List<double[]>();
            var labels = new List<int>();
            data.Add(new[] { -0.4326, 1.1909 });
            labels.Add(1);
            data.Add(new[] { 3.0, 4.0 });
            labels.Add(1);
            data.Add(new[] { 0.1253, -0.0376 });
            labels.Add(1);
            data.Add(new[] { 0.2877, 0.3273 });
            labels.Add(1);
            data.Add(new[] { -1.1465, 0.1746 });
            labels.Add(1);
            data.Add(new[] { 1.8133, 1.0139 });
            labels.Add(0);
            data.Add(new[] { 2.7258, 1.0668 });
            labels.Add(0);
            data.Add(new[] { 1.4117, 0.5593 });
            labels.Add(0);
            data.Add(new[] { 4.1832, 0.3044 });
            labels.Add(0);
            data.Add(new[] { 1.8636, 0.1677 });
            labels.Add(0);
            data.Add(new[] { 0.5, 3.2 });
            labels.Add(1);
            data.Add(new[] { 0.8, 3.2 });
            labels.Add(1);
            data.Add(new[] { 1.0, -2.2 });
            labels.Add(1);
            var n = labels.Count;

            var trainer = new SgdTrainer<double>(net) { LearningRate = 0.01, L2Decay = 0.001, BatchSize = n };

            // Training
            do
            {
                Classify2DUpdate(n, data, trainer, labels);
            } while (!Console.KeyAvailable);

            // Testing
            var netx = BuilderInstance.Volume.From(new double[2 * n], new Shape(1, 1, 2, n));
            for (var ix = 0; ix < n; ix++)
            {
                netx.Set(0, 0, 0, ix, data[ix][0]);
                netx.Set(0, 0, 1, ix, data[ix][1]);
            }

            var result = net.Forward(netx);
            var c = net.GetPrediction();
            var accurate = c[0] == labels[0];
        }

        private static void Classify2DUpdate(int n, List<double[]> data, TrainerBase<double> trainer, List<int> labels)
        {
            var avloss = 0.0;

            var netx = BuilderInstance.Volume.SameAs(new Shape(1, 1, 2, n));
            var hotLabels = BuilderInstance.Volume.SameAs(new Shape(1, 1, 2, n));

            for (var ix = 0; ix < n; ix++)
            {
                hotLabels.Set(0, 0, labels[ix], ix, 1.0);

                netx.Set(0, 0, 0, ix, data[ix][0]);
                netx.Set(0, 0, 1, ix, data[ix][1]);
            }

            for (var iters = 0; iters < 50; iters++)
            {
                trainer.Train(netx, hotLabels);
                avloss += trainer.Loss;
            }

            avloss /= 50.0;
            Console.WriteLine(" Loss:" + avloss);
        }

        private static void Main(string[] args)
        {
            Classify2DDemo();
        }
    }
}