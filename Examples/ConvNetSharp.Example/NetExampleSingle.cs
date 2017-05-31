using System.Windows;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Flow;
using ConvNetSharp.Utils.GraphVisualizer;

namespace ConvNetSharp.Example
{
    internal static class NetExampleSingle
    {
        public static void Example1()
        {
            var net = new Net<double>();
            var inputLayer = new InputLayer<double>();
            net.AddLayer(inputLayer);
            net.AddLayer(new FullyConnLayer<double>(6));
            net.AddLayer(new TanhLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(2));
            net.AddLayer(new TanhLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(2));
            net.AddLayer(new SoftmaxLayer<double>());

            var fun = net.Build();
            
            // Graph creation
            var x = ConvNetSharp<double>.PlaceHolder("x");
            var y = ConvNetSharp<double>.PlaceHolder("y");

            //  var cost = (fun - y) * (fun - y);

            var cost = -y * ConvNetSharp<double>.Log(x);

            var optimizer = new GradientDescentOptimizer<double>(0.01);

            using (var session = new Session<double>())
            {
                session.Differentiate(fun); // computes dCost/dW at every node of the graph

                // Display grpah
                var vm = new ViewModel<double>(fun);
                var app = new Application();
                app.Run(new GraphControl {DataContext = vm});
            }
        }
    }
}