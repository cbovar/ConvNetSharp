using System;
using System.Collections.Generic;
using System.Windows;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Utils.GraphVisualizer;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Example
{
    using cns = ConvNetSharp<float>;

    internal static class ExampleCpuSingle
    {
        public static void Example1()
        {
            // Graph creation
            var x = cns.PlaceHolder("x");
            var y = cns.PlaceHolder("y");

            var W = cns.Variable(1.0f, "W");
            var b = cns.Variable(2.0f, "b");

            var fun = x * W + b;

            var cost = (fun - y) * (fun - y);

            var optimizer = new GradientDescentOptimizer<float>(0.01f);

            using (var session = new Session<float>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                float currentCost;
                do
                {
                    var dico = new Dictionary<string, Volume<float>> {{"x", -2.0f}, {"y", 1.0f}};

                    currentCost = session.Run(cost, dico);
                    Console.WriteLine($"cost: {currentCost}");

                    var result = session.Run(fun, dico);
                    session.Run(optimizer, dico);
                } while (currentCost > 1e-5);


                float finalW = W.V;
                float finalb = b.V;
                Console.WriteLine($"fun = x * {finalW} + {finalb}");
              
                // Display grpah
                var vm = new ViewModel<float>(session);
                var app = new Application();
                app.Run(new GraphControl {DataContext = vm});
            }

            Console.ReadKey();
        }
    }
}