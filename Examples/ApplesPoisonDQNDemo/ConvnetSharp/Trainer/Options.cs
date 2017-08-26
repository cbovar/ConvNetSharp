using System;

namespace ConvnetSharpOLD
{
    [Serializable]
    public class Options
    {
        public string method = string.Empty;
        public int batchSize = int.MinValue;

        public double learningRate = double.MinValue;
        public double l1_decay = double.MinValue;
        public double l2_decay = double.MinValue;
        public double momentum = double.MinValue;
        public double beta1 = double.MinValue;
        public double beta2 = double.MinValue;
        public double ro = double.MinValue;
        public double eps = double.MinValue;
    }
}
