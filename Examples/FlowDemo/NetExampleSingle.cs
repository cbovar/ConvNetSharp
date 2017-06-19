using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Layers;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Utils.GraphVisualizer;
using ConvNetSharp.Volume;

namespace FlowDemo
{
    internal static class NetExampleSingle
    {
        private static int k;

        private static void Classify2DUpdate(int n, List<Volume<double>> data, Session<double> session, Op<double> fun, Op<double> cost, Op<double> optimizer, List<int> labels)
        {
            var avloss = 0.0;
            var netx = BuilderInstance<double>.Volume.SameAs(new Shape(1, 1, 2, n));
            var hotLabels = BuilderInstance<double>.Volume.SameAs(new Shape(1, 1, 2, n));

            for (var ix = 0; ix < n; ix++)
            {
                netx.Set(0, 0, 0, ix, data[ix].Get(0));
                netx.Set(0, 0, 1, ix, data[ix].Get(1));

                hotLabels.Set(0, 0, labels[ix], ix, 1.0);
            }

            var dico = new Dictionary<string, Volume<double>>
            {
                ["Y"] = hotLabels,
                ["input"] = netx
            };

            for (var iters = 0; iters < 50; iters++)
            {
                var currentCost = session.Run(cost, dico).ToArray().Average();

                var result = session.Run(fun, dico);

                //session.Dump(fun, "Flow.txt");

                session.Run(optimizer, dico);

               // session.Dump(fun, "Flow.txt");

                avloss += currentCost;
            }

            avloss /= 50.0;
            Console.WriteLine(k++ +" Loss:" + avloss);
        }

        public static void Example1()
        {
            #region Net

            var net = new Net<double>();
            net.AddLayer(new InputLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(6));
            net.AddLayer(new TanhLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(2));
            net.AddLayer(new TanhLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(2));
            var softmaxLayer = new SoftmaxLayer<double>();
            net.AddLayer(softmaxLayer);

            var fun = net.Build();

            var cost = softmaxLayer.Cost;

            #endregion

            #region Data

            // Data
            var data = new List<Volume<double>>();
            var labels = new List<int>();
            data.Add(new[] { -0.4326, 1.1909 });
            labels.Add(1);
            data.Add(new[] { 3.0, 4.0 });
            labels.Add(1);
            data.Add(new[] { 0.1253, -0.0376 });
            labels.Add(1);
            data.Add(new[] { 0.2877, 0.3273 });
            labels.Add(1);
            data.Add(new[] { -1.1465, 0.1746 });
            labels.Add(1);
            data.Add(new[] { 1.8133, 1.0139 });
            labels.Add(0);
            data.Add(new[] { 2.7258, 1.0668 });
            labels.Add(0);
            data.Add(new[] { 1.4117, 0.5593 });
            labels.Add(0);
            data.Add(new[] { 4.1832, 0.3044 });
            labels.Add(0);
            data.Add(new[] { 1.8636, 0.1677 });
            labels.Add(0);
            data.Add(new[] { 0.5, 3.2 });
            labels.Add(1);
            data.Add(new[] { 0.8, 3.2 });
            labels.Add(1);
            data.Add(new[] { 1.0, -2.2 });
            labels.Add(1);
            var n = labels.Count;

            #endregion

            using (var session = new Session<double>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                var optimizer = new GradientDescentOptimizer<double>(0.01);

                do
                {
                    Classify2DUpdate(n, data, session, fun, cost, optimizer, labels);
                } while (!Console.KeyAvailable);

                // Display graph
                var vm = new ViewModel<double>(cost);
                var app = new Application();
                app.Run(new GraphControl { DataContext = vm });
            }
        }
    }
}