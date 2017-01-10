using System;
using ConvNetSharp;
using ConvNetSharp.Layers;
using ConvNetSharp.Training;

namespace Regression1DDemo
{
    internal static class Program
    {
        private static void Regression1DDemo()
        {
            var net = new Net();
            net.AddLayer(new InputLayer(1, 1, 1));
            net.AddLayer(new FullyConnLayer(20));
            net.AddLayer(new ReluLayer());
            net.AddLayer(new FullyConnLayer(20));
            net.AddLayer(new SigmoidLayer());
            net.AddLayer(new FullyConnLayer(1));
            net.AddLayer(new RegressionLayer());

            var trainer = new SgdTrainer(net) { LearningRate = 0.01, Momentum = 0.0, BatchSize = 1, L2Decay = 0.001 };

            // Function we want to learn
            double[] x = { 0.0, 0.5, 1.0 };
            double[] y = { 0.0, 0.1, 0.2 };
            var n = x.Length;

            // Training
            do
            {
                RegressionUpdate(n, x, trainer, y);
            } while (!Console.KeyAvailable);

            // Testing
            var netx = new Volume(1, 1, 1);
            for (var ix = 0; ix < n; ix++)
            {
                netx.Set(0, 0, 0, x[ix]);
                var result = net.Forward(netx);
            }
        }

        private static void RegressionUpdate(int n, double[] x, TrainerBase trainer, double[] y)
        {
            var netx = new Volume(1, 1, 1);
            var avloss = 0.0;

            for (var iters = 0; iters < 50; iters++)
            {
                for (var ix = 0; ix < n; ix++)
                {
                    netx.Set(0, 0, 0, x[ix]);
                    trainer.Train(netx, y[ix]);
                    avloss += trainer.Loss;
                }
            }

            avloss /= n * 50.0;
            Console.WriteLine("Loss:" + avloss);
        }

        private static void Main(string[] args)
        {
            Regression1DDemo();
        }
    }
}