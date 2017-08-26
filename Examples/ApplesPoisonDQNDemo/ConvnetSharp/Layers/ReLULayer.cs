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
    public class ReLULayer : LayerBase
    {
        Util util = new Util();

        public ReLULayer(LayerDefinition def) : base()
        {
            // computed
            this.out_sx = def.in_sx;
            this.out_sy = def.in_sy;
            this.out_depth = def.in_depth;
            this.type = "relu";
        }

        public override Volume forward(Volume V, bool is_training)
        {
            this.in_act = V;
            var V2 = V.clone();
            var N = V.w.Length;
            var V2w = V2.w;
            for (var i = 0; i < N; i++)
            {
                if (V2w[i] < 0) V2w[i] = 0; // threshold at 0
            }
            this.out_act = V2;
            return this.out_act;
        }

        public override double backward(object y)
        {
            var V = this.in_act; // we need to set dw of this
            var V2 = this.out_act;
            var N = V.w.Length;
            V.dw = util.zeros(N); // zero out gradient wrt data
            for (var i = 0; i < N; i++)
            {
                if (V2.w[i] <= 0) V.dw[i] = 0; // threshold
                else V.dw[i] = V2.dw[i];
            }

            return 0.0;
        }
        public override Gradient[] getParamsAndGrads()
        {
            return new List<Gradient>().ToArray();
        }
    }
}
