using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvnetSharpOLD
{
    [Serializable]
    public class InputLayer : LayerBase
    {
        Util util = new Util();

        public InputLayer(LayerDefinition def) : base()
        {
            // required: depth
            this.out_depth = def.out_depth;

            // optional: default these dimensions to 1
            this.out_sx = def.out_sx;
            this.out_sy = def.out_sy;

            // computed
            this.type = "input";
        }

        public override Volume forward(Volume V, bool is_training)
        {
            this.in_act = V;
            this.out_act = V;
            return this.out_act; // simply identity function for now
        }

        public override double backward(object y) { return 0.0; }
        public override Gradient[] getParamsAndGrads()
        {
            return new List<Gradient>().ToArray();
        }
    }
}
