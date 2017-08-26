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
    public class RegressionLayer : LayerBase
    {
        Util util = new Util();

        public RegressionLayer(LayerDefinition def) : base()
        {
            // computed
            this.num_inputs = def.in_sx * def.in_sy * def.in_depth;
            this.out_depth = this.num_inputs;
            this.out_sx = 1;
            this.out_sy = 1;
            this.type = "regression";
        }

        public override Volume forward(Volume V, bool is_training)
        {
            this.in_act = V;
            this.out_act = V;
            return this.out_act; // simply identity function for now
        }

        // y is a list here of size num_inputs
        // or it can be a number if only one value is regressed
        // or it can be a struct {dim: i, val: x} where we only want to 
        // regress on dimension i and asking it to have value x
        public override double backward(object y)
        {
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.in_act;
            x.dw = util.zeros(x.w.Length); // zero out the gradient of input Vol
            var loss = 0.0;
            if (y.GetType().Equals(typeof(Array))) {

                var Y = (double[])y;

                for (var i = 0; i < this.out_depth; i++)
                {
                    var dy = x.w[i] - Y[i];
                    x.dw[i] = dy;
                    loss += 0.5 * dy * dy;
                }
            }
            else if (y.GetType().Equals(typeof(Double)))
            {
                // lets hope that only one number is being regressed
                var dy = x.w[0] - (double)y;
                x.dw[0] = dy;
                loss += 0.5 * dy * dy;
            }
            else
            {
                // assume it is a struct with entries .dim and .val
                // and we pass gradient only along dimension dim to be equal to val
                var i = ((Entry)y).dim;
                var yi = ((Entry)y).val;
                var dy = x.w[i] - yi;
                x.dw[i] = dy;
                loss += 0.5 * dy * dy;
            }

            return loss;
        }

        public override Gradient[] getParamsAndGrads()
        {
            return new List<Gradient>().ToArray();
        }
    }
}
