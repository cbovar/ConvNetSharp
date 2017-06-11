using System;
using System.Collections.Generic;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Example
{
    using cns = ConvNetSharp<double>;

    internal static class ExampleCpuDouble
    {
        public static void Example1()
        {
            // Graph creation
            var x = cns.PlaceHolder("x");
            var y = cns.PlaceHolder("y");

            var W = cns.Variable(1.0, "W");
            var b = cns.Variable(2.0, "b");

            var fun = x * W + b;

            var cost = (fun - y) * (fun - y);

            var optimizer = new GradientDescentOptimizer<double>(learningRate: 0.01);

            using (var session = new Session<double>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                double currentCost;
                do
                {
                    var dico = new Dictionary<string, Volume<double>> { { "x", -2.0 }, { "y", 1.0f } };

                    currentCost = session.Run(cost, dico);
                    Console.WriteLine($"cost: {currentCost}");

                    var result = session.Run(fun, dico);
                    session.Run(optimizer, dico);
                } while (currentCost > 1e-5);
            }

            double finalW = W.V;
            double finalb = b.V;
            Console.WriteLine($"fun = x * {finalW} + {finalb}");
            Console.ReadLine();
        }
    }
}