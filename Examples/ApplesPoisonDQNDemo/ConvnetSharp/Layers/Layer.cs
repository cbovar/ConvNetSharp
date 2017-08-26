using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvnetSharpOLD
{
    [Serializable]
    public abstract class LayerBase
    {
        public string type;
        public string activation;
        public int group_size;
        public int num_neurons;
        public int num_classes;
        public int num_inputs;
        public double bias_pref;
        public double drop_prob;

        public int out_depth;
        public int out_sx;
        public int out_sy;
        public int in_depth;
        public int in_sx;
        public int in_sy;
        public int sx;
        public int sy;

        public Volume in_act;
        public Volume out_act;

        public double l1_decay_mul;
        public double l2_decay_mul;

        public List<Volume> filters;

        public abstract Gradient[] getParamsAndGrads();
        public abstract Volume forward(Volume V, bool is_training);
        public abstract double backward(object y);
    }
}
