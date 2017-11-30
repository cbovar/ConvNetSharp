using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Utils.GraphVisualizer;
using ConvNetSharp.Volume;

namespace FlowDemo
{
    internal static class ExampleCpuDouble
    {
        /// <summary>
        /// Solves y = x * W + b (CPU double version)
        /// for y = 1 and x = -2
        /// </summary>
        public static void Example1()
        {
            var cns = ConvNetSharp<double>.Instance;

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

            double finalW = W.Result;
            double finalb = b.Result;
            Console.WriteLine($"fun = x * {finalW} + {finalb}");
            Console.ReadLine();
        }

        public static void Example2()
        {
            var cns = ConvNetSharp<double>.Instance;

            // Graph creation
            var x = cns.PlaceHolder("x");
            var y = cns.PlaceHolder("y");

            var b = cns.Variable(2.0, "b");

            var fun = x + b;

            var cost = (y - fun) * (y - fun);

            var optimizer = new GradientDescentOptimizer<double>(learningRate: 0.01);

            using (var session = new Session<double>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                double currentCost;
                do
                {
                    var xx = BuilderInstance<double>.Volume.From(new[] { -2.0, -3.0, -10.0 }, new Shape(1, 1, 1, 3));
                    var yy = BuilderInstance<double>.Volume.From(new[] { -5.0, -6.0, -13.0 }, new Shape(1, 1, 1, 3));

                    var dico = new Dictionary<string, Volume<double>> { { "x", xx }, { "y", yy } };

                    currentCost = Math.Abs(session.Run(cost, dico).ToArray().Sum());
                    Console.WriteLine($"cost: {currentCost}");

                    var result = session.Run(fun, dico);
                    session.Run(optimizer, dico);
                } while (currentCost > 1e-5);
            }

            // Display derivate at b
            var vm = new ViewModel<double>(b.Derivate);
            var app = new Application();
            app.Run(new GraphControl { DataContext = vm });

            double finalb = b.Result;
            Console.WriteLine($"fun = x + {finalb}");
            Console.ReadLine();
        }
    }
}