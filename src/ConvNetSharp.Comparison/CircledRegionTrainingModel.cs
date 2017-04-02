using System;
using System.Collections.Generic;

namespace ConvNetSharp.Comparison
{
    public class CircledRegionTrainingModel
    {
        public readonly List<double[]> Inputs;
        public readonly List<double[]> Outputs;

        public int NmInputs => 2;
        public int NmOutputs => 3;

        public CircledRegionTrainingModel()
        {
            var random = new System.Random(1234);

            this.Inputs = new List<double[]>();
            this.Outputs = new List<double[]>();

            Func<double> getNum = () => random.NextDouble()*20.0 - 10.0;

            for (var i = 0; i < 1000; i++)
            {
                var x = getNum();
                var y = getNum();

                var input = new double[2];
                input[0] = x;
                input[1] = y;
                var output = new[] {0.0, 0.0, 0.0};
                if (Math.Sqrt(x*x + y*y) > 6)
                    output[0] = 1;
                else if (Math.Sqrt(x*x + y*y) > 3)
                    output[1] = 1;
                else
                    output[2] = 1;
                Inputs.Add(input);
                Outputs.Add(output);
            }
        }
    }
}