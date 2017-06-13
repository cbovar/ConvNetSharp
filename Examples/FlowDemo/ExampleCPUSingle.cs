using System;
using System.Collections.Generic;
using System.Windows;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Utils.GraphVisualizer;
using ConvNetSharp.Volume;

namespace FlowDemo
{
    internal static class ExampleCpuSingle
    {
        public static void Example1()
        {
            var cns = ConvNetSharp<float>.Instance;

            // Graph creation
            var x = cns.PlaceHolder("x");
            var y = cns.PlaceHolder("y");

            var W = cns.Variable(1.0f, "W");
            var b = cns.Variable(2.0f, "b");

            var fun = x * W + b + cns.Sigmoid(x);

            var cost = (fun - y) * (fun - y);

            var optimizer = new GradientDescentOptimizer<float>(0.01f);

            using (var session = new Session<float>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                float currentCost;
                do
                {
                    var dico = new Dictionary<string, Volume<float>> { { "x", -2.0f }, { "y", 1.0f } };

                    currentCost = session.Run(cost, dico);
                    Console.WriteLine($"cost: {currentCost}");

                    var result = session.Run(fun, dico);
                    session.Run(optimizer, dico);
                } while (currentCost > 1e-5);


                float finalW = W.V;
                float finalb = b.V;
                Console.WriteLine($"fun = x * {finalW} + {finalb}");

                // Display grpah
                var vm = new ViewModel<float>(cost);
                var app = new Application();
                app.Run(new GraphControl { DataContext = vm });
            }

            Console.ReadKey();
        }

        public static void Example2()
        {
            var cns = ConvNetSharp<float>.Instance;

            // Graph creation
            var x = cns.PlaceHolder("x");
            var fun = cns.Const(2,"2") * x;

            using (var session = new Session<float>())
            {
                session.Differentiate(fun); // computes dCost/dW at every node of the graph

                //var dico = new Dictionary<string, Volume<float>> { { "x", -10.0f }};
                //var result = session.Run(fun, dico);

                // Display grpah
                var vm = new ViewModel<float>(x.Derivate);
                var app = new Application();
                app.Run(new GraphControl { DataContext = vm });
            }

            Console.ReadKey();
        }
    }
}