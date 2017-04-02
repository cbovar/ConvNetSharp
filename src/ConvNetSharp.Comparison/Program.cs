extern alias previous;
extern alias latest;

using System;
using System.Diagnostics;

using A = previous.ConvNetSharp;
using B = latest.ConvNetSharp.Core;
using BVolume = latest.ConvNetSharp.Volume;

namespace ConvNetSharp.Comparison
{
    public class Program
    {
        public static void Main(string[] args)
        {
            //var model = new XorTrainingModel();
            var model = new CircledRegionTrainingModel();

            const int BATCH_SIZE = 10;
            const double LEARN_RATE = 0.01;
            const double MOMENTUM = 0.05;
            
            var aNet = CreateOldNet(model.NmInputs, model.NmOutputs);
            var bNet = CreateNewNet(model.NmInputs, model.NmOutputs);
            var builder = BVolume.BuilderInstance<double>.Volume;

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
            var bIn = builder.SameAs(new BVolume.Shape(1, 1, model.NmInputs, BATCH_SIZE));
            var bOut = builder.SameAs(new BVolume.Shape(1, 1, model.NmOutputs, BATCH_SIZE));
            for (var epoch = 0; epoch < 50; epoch++)
            {
                for (var i = 0; i < model.Inputs.Count; i++)
                {
                    var input = model.Inputs[i];
                    var output = model.Outputs[i];

                    //convert input and output to what trainer expects
                    var aIn = new A.Volume(1, 1, model.NmInputs);
                    for (var x = 0; x < input.Length; x++)
                        aIn.Set(x, input[x]);
                    var aOut = Array.IndexOf(output, 1.0);

                    //trainer A takes care of batching inside
                    //expects a class number, not softmax array
                    aTrainer.Train(aIn, aOut);

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
                    }
                }

                //write A vs B losses
                Console.WriteLine(
                    $"{epoch:000}       A == {aTrainer.Loss:0.000} vs B == {bTrainer.Loss:0.000} ----> DIFF {aTrainer.Loss - bTrainer.Loss: 0.000}");
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