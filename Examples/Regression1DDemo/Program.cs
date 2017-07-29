using System;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;

namespace Regression1DDemo
{
    internal class Program
    {
        private static void Main()
        {
            Regression1DDemo();
        }

        private static void Regression1DDemo()
        {
            var net = new Net<double>();
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
            var netx = BuilderInstance.Volume.SameAs(new Shape(1, 1, 1));
            for (var ix = 0; ix < n; ix++)
            {
                netx.Set(0, 0, 0, x[ix]);
                var result = net.Forward(netx);
            }
        }

        private static void RegressionUpdate(int n, double[] x, TrainerBase<double> trainer, double[] y)
        {
            var netx = BuilderInstance.Volume.SameAs(new Shape(1, 1, 1, n));
            var nety = BuilderInstance.Volume.SameAs(new Shape(1, 1, 1, n));
            var avloss = 0.0;

            for (var ix = 0; ix < n; ix++)
            {
                netx.Set(0, 0, 0, ix, x[ix]);
                nety.Set(0, 0, 0, ix, y[ix]);
            }

            for (var iters = 0; iters < 50; iters++)
            {
                trainer.Train(netx, nety);
                avloss += trainer.Loss;
            }

            avloss /= n * 50.0;
            Console.WriteLine("Loss:" + avloss);
        }
    }
}