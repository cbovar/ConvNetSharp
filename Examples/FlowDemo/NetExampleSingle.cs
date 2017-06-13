using System;
using System.Collections.Generic;
using System.Windows;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Layers;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Utils.GraphVisualizer;
using ConvNetSharp.Volume;

namespace FlowDemo
{
    internal static class NetExampleSingle
    {
        public static void Example1()
        {
            var net = new Net<double>();
            net.AddLayer(new InputLayer<double>());
            //net.AddLayer(new FullyConnLayer<double>(6));
            //net.AddLayer(new TanhLayer<double>());
            //net.AddLayer(new FullyConnLayer<double>(2));
            //net.AddLayer(new TanhLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(2));
            var softmaxLayer = new SoftmaxLayer<double>();
            net.AddLayer(softmaxLayer);

            var fun = net.Build();

            var cost = softmaxLayer.Cost;

            var optimizer = new GradientDescentOptimizer<double>(0.01);

            using (var session = new Session<double>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                var y = ((Volume<double>)new[] { 0.0, 1.0 }).ReShape(1, 1, 2, 1);
                var dico = new Dictionary<string, Volume<double>> { { "input", -2.0 }, { "Y", y } };

                for (int i = 0; i < 1000; i++)
                {
                    var currentCost = session.Run(cost, dico);
                    Console.WriteLine($"cost: {currentCost}");

                    var result = session.Run(fun, dico);
                    session.Run(optimizer, dico);
                }

                // Display graph
                var vm = new ViewModel<double>(cost);
                var app = new Application();
                app.Run(new GraphControl { DataContext = vm });
            }
        }
    }
}