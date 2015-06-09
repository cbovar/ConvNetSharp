using System;
using System.Collections.Generic;
using System.IO;
using System.Xml;
using ConvNetSharp;

namespace DtwDemo
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //double[] s = { 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0 };
            //double[] t1 = { 0.0, 1.0, 2.0 };
            //double[] t2 = { 0.0, 0.0, 1.0, 2.0 };
            //double[] t3 = { 0.0, 0.0, 0.0, 1.0, 2.0 };
            //double[] t4 = { 0.0, 0.0, 0.0, 0.0, 1.0, 2.0 };
            //double[] t5 = { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0 };


            //var d1 = DynamicTimeWarpLayer.Dtw(s, t1, 2);
            //var d2 = DynamicTimeWarpLayer.Dtw(s, t2, 2);
            //var d3 = DynamicTimeWarpLayer.Dtw(s, t3, 2);
            //var d4 = DynamicTimeWarpLayer.Dtw(s, t4, 2);
            //var d5 = DynamicTimeWarpLayer.Dtw(s, t5, 2);

            List<double> data = new List<double>();
            using (var sr = new StreamReader(@"C:\pro\temp\test.csv"))
            {
                sr.ReadLine();//header

                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine().Split(',')[1];
                    data.Add(double.Parse(line));
                }
            }

            var net = new Net();
            net.AddLayer(new InputLayer(10, 1, 1));
            net.AddLayer(new DynamicTimeWarpLayer(3, 2));
            net.AddLayer(new RegressionLayer(1));


            //var input1 = new Volume(10, 1, 1, 0);
            //input1.Set(0, 0, 0, 0.0);
            //input1.Set(1, 0, 0, 1.0);
            //input1.Set(2, 0, 0, 2.0);
            //input1.Set(3, 0, 0, 1.0);
            //input1.Set(2, 0, 0, 0.0);

            //var input2 = new Volume(10, 1, 1, 0);
            //input2.Set(0, 0, 0, 2.0);
            //input2.Set(1, 0, 0, 1.0);
            //input2.Set(2, 0, 0, 0.0);
            //input2.Set(3, 0, 0, 1.0);
            //input2.Set(2, 0, 0, 2.0);


            var trainer = new Trainer(net) { LearningRate = 0.01, Momentum = 0.0, BatchSize = 10, L2Decay = 0.001 };

            //double[] input = new double[5];

            do
            {
                for (int i = 0; i < data.Count-6; i++)
                {
                    var vol = new Volume(5, 1, 1, 0);

                    for (int j = 0; j < 5; j++)
                    {
                        vol.Set(j, 0, 0, data[i + j]);
                        //input[j] = data[i + j];
                    }

                    var res1 = net.Forward(vol);
                    trainer.Train(vol, data[i + 5]);
                }



                
               

           //     var res1 = net.Forward(input1);
                //var res2 = net.Forward(input2);

                Console.WriteLine(trainer.CostLoss);

            } while (true);
        }
    }
}