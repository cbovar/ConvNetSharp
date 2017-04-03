extern alias previous;
extern alias latest;

using System;
using System.Diagnostics;
using A = previous.ConvNetSharp;
using B = latest.ConvNetSharp.Core;
using Shape = latest.ConvNetSharp.Volume.Shape;
using BVolume = latest.ConvNetSharp.Volume.Double;

namespace ConvNetSharp.Comparison
{
    public class Program
    {
        public static void Main(string[] args)
        {
            BVolume.BuilderInstance.Volume = new BVolume.VolumeBuilder();
            var builder = BVolume.BuilderInstance.Volume;

            var set = new XorTrainingSet();
            //var set = new CircledRegionTrainingSet();

            const int BATCH_SIZE = 1;
            const double LEARN_RATE = 0.05;
            const double MOMENTUM = 0;

            var aNet = CreateOldNet(set.NmInputs, set.NmOutputs);
            var bNet = CreateNewNet(set.NmInputs, set.NmOutputs);
           
            var aTrainer = new A.Training.SgdTrainer(aNet)
            {
                BatchSize = BATCH_SIZE,
                L1Decay = 0,
                L2Decay = 0,
                LearningRate = LEARN_RATE,
                Momentum = MOMENTUM
            };

            var bTrainer = new B.Training.SgdTrainer<double>(bNet)
            {
                BatchSize = BATCH_SIZE,
                L1Decay = 0,
                L2Decay = 0,
                LearningRate = LEARN_RATE,
                Momentum = MOMENTUM
            };

            int xIn = 0, xOut = 0;
            var bIn = builder.SameAs(new Shape(1, 1, set.NmInputs, BATCH_SIZE));
            var bOut = builder.SameAs(new Shape(1, 1, set.NmOutputs, BATCH_SIZE));

            var epoch = 0;
            var i = 0;
            double aLossSum = 0, bLossSum = 0;
            int aLossCount = 0, bLossCount = 0;
            while (epoch < 50)
            {
                var input = set.Inputs[i];
                var output = set.Outputs[i];

                //convert input and output to what trainer expects
                var aIn = new A.Volume(1, 1, set.NmInputs);
                for (var x = 0; x < input.Length; x++)
                    aIn.Set(x, input[x]);
                var aOut = Array.IndexOf(output, 1.0);

                //trainer A takes care of batching inside
                //expects a class number, not softmax array
                //pool loss, since loss is per sample, not per batch
                aTrainer.Train(aIn, aOut);
                aLossSum += aTrainer.CostLoss;
                aLossCount++;

                //pool input and output into batch for trainer B
                foreach (var v in input)
                    bIn.Set(xIn++, v);
                foreach (var v in output)
                    bOut.Set(xOut++, v);

                //when batch for B full, execute training
                if (xIn == bIn.Shape.TotalLength)
                {
                    Debug.Assert(xOut == bOut.Shape.TotalLength);
                    xIn = 0;
                    xOut = 0;
                    bTrainer.Train(bIn, bOut);
                    bLossSum += bTrainer.Loss;
                    bLossCount++;
                }

                i++;
                if (i >= set.Inputs.Count)
                {
                    //calculate loss per batch for aTrainer
                    var aLossAvg = aLossSum/aLossCount;
                    aLossSum = 0;
                    aLossCount = 0;
                    var bLossAvg = bLossSum/bLossCount;
                    bLossSum = 0;
                    bLossCount = 0;

                    var diff = Math.Abs(aLossAvg - bLossAvg);

                    //write A vs B losses
                    Console.WriteLine(
                        $"{epoch:000}       A == {aLossAvg:0.000} vs B == {bLossAvg:0.000} " +
                        $"----> DIFF {diff: 0.000}");
                    epoch++;
                    i = 0;
                }
            }
        }

        private static A.INet CreateOldNet(int nmInputs, int nmOutputs)
        {
            var net = new A.Net();
            net.AddLayer(new A.Layers.InputLayer(1, 1, nmInputs));
            net.AddLayer(new A.Layers.FullyConnLayer(10));
            net.AddLayer(new A.Layers.ReluLayer());
            net.AddLayer(new A.Layers.FullyConnLayer(5));
            net.AddLayer(new A.Layers.ReluLayer());
            net.AddLayer(new A.Layers.FullyConnLayer(nmOutputs));
            net.AddLayer(new A.Layers.SoftmaxLayer(nmOutputs));
            return net;
        }

        private static B.INet<double> CreateNewNet(int nmInputs, int nmOutputs)
        {
            var net = new B.Net<double>();
            net.AddLayer(new B.Layers.InputLayer<double>(1, 1, nmInputs));
            net.AddLayer(new B.Layers.FullyConnLayer<double>(10));
            net.AddLayer(new B.Layers.ReluLayer<double>());
            net.AddLayer(new B.Layers.FullyConnLayer<double>(5));
            net.AddLayer(new B.Layers.ReluLayer<double>());
            net.AddLayer(new B.Layers.FullyConnLayer<double>(nmOutputs));
            net.AddLayer(new B.Layers.SoftmaxLayer<double>(nmOutputs));
            return net;
        }
    }
}