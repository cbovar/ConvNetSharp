using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Flow.Serialization;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Utils.GraphVisualizer;
using ConvNetSharp.Volume;

namespace FlowDemo
{
    internal static class ExampleCpuSingle
    {
        /// <summary>
        /// Solves y = x * W + b (CPU single version)
        /// for y = 1 and x = -2
        /// 
        /// This also demonstrates how to save and load a graph
        /// </summary>
        public static void Example1()
        {
            var cns = ConvNetSharp<float>.Instance;

            // Graph creation
            Op<float> cost;
            Op<float> fun;
            if (File.Exists("test.graphml"))
            {
                Console.WriteLine("Loading graph from disk.");
                var ops = SerializationExtensions.Load<float>("test", true);

                fun = ops[0];
                cost = ops[1];
            }
            else
            {
                var x = cns.PlaceHolder("x");
                var y = cns.PlaceHolder("y");

                var W = cns.Variable(1.0f, "W");
                var b = cns.Variable(2.0f, "b");

                fun = x * W + b;

                cost = (fun - y) * (fun - y);
            }


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

                float finalW = session.GetVariableByName(fun, "W").Result;
                float finalb = session.GetVariableByName(fun, "b").Result;
                Console.WriteLine($"fun = x * {finalW} + {finalb}");

                fun.Save("test", cost);

                // Display grpah
                var vm = new ViewModel<float>(cost);
                var app = new Application();
                app.Run(new GraphControl { DataContext = vm });
            }

            Console.ReadKey();
        }

        /// <summary>
        /// Computes and displays d(f(x))/dx = d(2x)/x
        /// </summary>
        public static void Example2()
        {
            var cns = ConvNetSharp<float>.Instance;

            // Graph creation
            var x = cns.PlaceHolder("x");
            var fun = cns.Const(2, "2") * x;

            using (var session = new Session<float>())
            {
                session.Differentiate(fun); // computes dCost/dW at every node of the graph

                // Display grpah
                var vm = new ViewModel<float>(x.Derivate);
                var app = new Application();
                app.Run(new GraphControl { DataContext = vm });
            }

            Console.ReadKey();
        }
    }
}