using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Example
{
    using cns = ConvNetSharp<float>;
    using Volume.GPU.Single;

    public class ExampleGpuSingle
    {
        public static void Example1()
        {
            BuilderInstance<float>.Volume = new VolumeBuilder();

            // Graph creation
            var x = cns.PlaceHolder("x");
            var y = cns.PlaceHolder("y");

            var W = cns.Variable(1.0f, "W");
            var b = cns.Variable(2.0f, "b");

            var fun = x * W + b;

            var cost = (fun - y) * (fun - y);

            var optimizer = new GradientDescentOptimizer<float>(learningRate: 0.01f);

            using (var session = new Session<float>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                double currentCost;
                do
                {
                    var dico = new Dictionary<string, Volume<float>> { { "x", -2.0f }, { "y", 1.0f } };

                    currentCost = session.Run(cost, dico);
                    Console.WriteLine($"cost: {currentCost}");

                    var result = session.Run(fun, dico);
                    session.Run(optimizer, dico);
                } while (currentCost > 1e-5);
            }

            double finalW = W.V;
            double finalb = b.V;
            Console.WriteLine($"fun = x * {finalW} + {finalb}");
            Console.ReadKey();
        }
    }
}