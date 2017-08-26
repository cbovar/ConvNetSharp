using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvnetSharpOLD
{
    [Serializable]
    public class LayerDefinition
    {
        public string type;
        public string activation;
        public int group_size = int.MinValue;
        public int num_neurons = int.MinValue;
        public int num_classes = int.MinValue;        
        public int num_inputs = int.MinValue;
        public double bias_pref = double.MinValue;
        public double drop_prob = double.MinValue;

        public int out_depth = int.MinValue;
        public int out_sx = int.MinValue;
        public int out_sy = int.MinValue;
        public int in_depth = int.MinValue;
        public int in_sx = int.MinValue;
        public int in_sy = int.MinValue;
        public int sx = int.MinValue;
        public int sy = int.MinValue;
        
        public double l1_decay_mul = double.MinValue;
        public double l2_decay_mul = double.MinValue;

        public List<Volume> filters;
        public int n_filters = int.MinValue;
        public int stride = int.MinValue;
        public int pad = int.MinValue;
    }
}
