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
            var graph = new ConvNetSharp<float>();

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
                var x = graph.PlaceHolder("x");
                var y = graph.PlaceHolder("y");

                var W = graph.Variable(1.0f, "W", true);
                var b = graph.Variable(2.0f, "b", true);

                fun = x * W + b;

                cost = (fun - y) * (fun - y);
            }


            var optimizer = new AdamOptimizer<float>(graph, 0.01f, 0.9f, 0.999f, 1e-08f);

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

                // Display graph
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
            var graph = new ConvNetSharp<float>();

            // Graph creation
            var x = graph.PlaceHolder("x");
            var fun = 2.0f * x;

            using (var session = new Session<float>())
            {
                session.Differentiate(fun); // computes dCost/dW at every node of the graph

                // Display graph
                var vm = new ViewModel<float>(x.Derivate);
                var app = new Application();
                app.Run(new GraphControl { DataContext = vm });
            }

            Console.ReadKey();
        }

        /// <summary>
        /// Computes and displays t = t + 1
        /// </summary>
        public static void Example3()
        {
            var graph = new ConvNetSharp<float>();

            // Graph creation
            var t = graph.PlaceHolder("t");
            var fun = graph.Assign(t, t + 1);

            using (var session = new Session<float>())
            {
                session.InitializePlaceHolders(fun, new Dictionary<string, Volume<float>> { { "t", 1.0f } });

                do
                {
                    session.Run(fun, null);

                    var x  = t.Result.Get(0);
                    Console.WriteLine(x);

                } while (!Console.KeyAvailable);

                // Display graph
                var vm = new ViewModel<float>(fun);
                var app = new Application();
                app.Run(new GraphControl { DataContext = vm });
            }
        }
    }
}