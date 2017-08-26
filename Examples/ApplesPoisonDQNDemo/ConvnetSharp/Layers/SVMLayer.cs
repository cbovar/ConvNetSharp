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
    public class SVMLayer : LayerBase
    {
        Util util = new Util();

        public SVMLayer(LayerDefinition def) : base()
        {
            // computed
            this.num_inputs = def.in_sx * def.in_sy * def.in_depth;
            this.out_depth = this.num_inputs;
            this.out_sx = 1;
            this.out_sy = 1;
            this.type = "svm";
        }

        public override Volume forward(Volume V, bool is_training)
        {
            this.in_act = V;
            this.out_act = V; // nothing to do, output raw scores
            return V;
        }

        public override double backward(object y)
        {
            var index = (int)y;

            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.in_act;
            x.dw = util.zeros(x.w.Length); // zero out the gradient of input Vol

            // we're using structured loss here, which means that the score
            // of the ground truth should be higher than the score of any other 
            // class, by a margin
            var yscore = x.w[index]; // score of ground truth
            var margin = 1.0;
            var loss = 0.0;
            for (var i = 0; i < this.out_depth; i++)
            {
                if (index == i) { continue; }
                var ydiff = -yscore + x.w[i] + margin;
                if (ydiff > 0)
                {
                    // violating dimension, apply loss
                    x.dw[i] += 1;
                    x.dw[index] -= 1;
                    loss += ydiff;
                }
            }

            return loss;
        }

        public override Gradient[] getParamsAndGrads()
        {
            return new List<Gradient>().ToArray();
        }
    }
}
