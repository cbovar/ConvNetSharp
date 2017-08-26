using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvnetSharpOLD
{
    [Serializable]
    public class Gradient
    {
        public double[] w;
        public double[] dw;
        public double l1_decay_mul = double.MinValue;
        public double l2_decay_mul = double.MinValue;
    }
}
