using System;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Serialization;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;

namespace SaveAndLoadDemo
{
    internal class Program
    {
        /// <summary>
        ///     This sample shows how to serialize and deserialize a ConvNetSharp.Core network
        ///     1) Network creation
        ///     2) Dummy Training (only use a single data point)
        ///     3) Serialization
        ///     4) Deserialization
        /// </summary>
        private static void Main()
        {
            // 1) Network creation
            var net = new Net<double>();

            net.AddLayer(new InputLayer(1, 1, 2));
            net.AddLayer(new FullyConnLayer(20));
            net.AddLayer(new ReluLayer());
            net.AddLayer(new FullyConnLayer(10));
            net.AddLayer(new SoftmaxLayer(10));

            // 2) Dummy Training (only use a single data point)
            var x = BuilderInstance.Volume.From(new[] {0.3, -0.5}, new Shape(2));
            var y = BuilderInstance.Volume.From(new[] {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new Shape(10));

            var count = 0;
            var trainer = new SgdTrainer(net) {LearningRate = 0.01};
            do
            {
                trainer.Train(x, y); // train the network, specifying that x is class zero
                Console.WriteLine($"Loss: {trainer.Loss}");
                count++;
            } while (trainer.Loss > 1e-2);

            Console.WriteLine($"{count}");

            // Forward pass with original network
            var prob1 = net.Forward(x);
            Console.WriteLine("probability that x is class 0: " + prob1.Get(0));

            // 3) Serialization
            var json = net.ToJson();

            // 4) Deserialization
            var deserialized = SerializationExtensions.FromJson<double>(json);

            // Forward pass with deserialized network
            var prob2 = deserialized.Forward(x);
            Console.WriteLine("probability that x is class 0: " + prob2.Get(0)); // This should give exactly the same result as previous network evaluation

            Console.ReadLine();
        }
    }
}