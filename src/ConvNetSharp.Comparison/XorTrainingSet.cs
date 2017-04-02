using System;
using System.Collections.Generic;

namespace ConvNetSharp.Comparison
{
    public class XorTrainingSet
    {
        public readonly List<double[]> Inputs;
        public readonly List<double[]> Outputs;

        public int NmInputs => 2;
        public int NmOutputs => 2;

        public XorTrainingSet()
        {
            var random = new Random(1234);

            this.Inputs = new List<double[]>();
            this.Outputs = new List<double[]>();

            for (var i = 0; i < 1000; i++)
            {
                var x = random.NextDouble();
                var y = random.NextDouble();

                var input = new double[2];
                input[0] = x;
                input[1] = y;

                var xor = x > 0.5 ^ y > 0.5;
                var output = new[] {0.0, 0.0};
                if (xor)
                    output[1] = 1;
                else
                    output[0] = 1;

                Inputs.Add(input);
                Outputs.Add(output);
            }
        }
    }
}