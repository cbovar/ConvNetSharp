using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Flow.Serialization;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Volume;

namespace SaveAndLoadDemo.Flow
{
    internal class Program
    {
        /// <summary>
        ///     This sample shows how to serialize and deserialize a ConvNetSharp.Flow network
        ///     1) Graph creation
        ///     2) Dummy Training (only use a single data point)
        ///     3) Serialization
        ///     4) Deserialization
        /// </summary>
        private static void Main()
        {
            var cns = new ConvNetSharp<double>();

            // 1) Graph creation
            var input = cns.PlaceHolder("x"); // input

            var dense1 = cns.Dense(input, 20) + cns.Variable(BuilderInstance<double>.Volume.From(new double[20].Populate(0.1), new Shape(20)), "bias1", true);
            var relu = cns.Relu(dense1);
            var dense2 = cns.Dense(relu, 10) + cns.Variable(new Shape(10), "bias2", true);
            var softmax = cns.Softmax(dense2); // output

            var output = cns.PlaceHolder("y"); // ground truth
            var cost = new SoftmaxCrossEntropy<double>(cns, softmax, output);

            var x = BuilderInstance<double>.Volume.From(new[] {0.3, -0.5}, new Shape(2));
            var y = BuilderInstance<double>.Volume.From(new[] {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new Shape(10));
            var dico = new Dictionary<string, Volume<double>> {{"x", x}, {"y", y}};

            var count = 0;
            var optimizer = new GradientDescentOptimizer<double>(cns, 0.01);
            using (var session = new Session<double>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                // 2) Dummy Training (only use a single data point)
                double currentCost;
                do
                {
                    currentCost = Math.Abs(session.Run(cost, dico, false).ToArray().Sum());
                    Console.WriteLine($"cost: {currentCost}");

                    session.Run(optimizer, dico);
                    count++;
                } while (currentCost > 1e-2);

                Console.WriteLine($"{count}");

                // Forward pass with original network
                var result = session.Run(softmax, new Dictionary<string, Volume<double>> {{"x", x}});
                Console.WriteLine("probability that x is class 0: " + result.Get(0));
            }

            // 3) Serialization
            softmax.Save("MyNetwork");

            // 4) Deserialization
            var deserialized = SerializationExtensions.Load<double>("MyNetwork", false)[0]; // first element is the model (second element is the cost if it was saved along)

            using (var session = new Session<double>())
            {
                // Forward pass with deserialized network
                var result = session.Run(deserialized, new Dictionary<string, Volume<double>> {{"x", x}});
                Console.WriteLine("probability that x is class 0: " + result.Get(0)); // This should give exactly the same result as previous network evaluation
            }

            Console.ReadLine();
        }
    }
}