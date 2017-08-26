using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvnetSharpOLD
{
    // An inefficient dropout layer
    // Note this is not most efficient implementation since the layer before
    // computed all these activations and now we're just going to drop them :(
    // same goes for backward pass. Also, if we wanted to be efficient at test time
    // we could equivalently be clever and upscale during train and copy pointers during test
    // todo: make more efficient.
    [Serializable]
    public class DropoutLayer : LayerBase
    {
        bool[] dropped;

        Util util = new Util();

        public DropoutLayer(LayerDefinition def) : base()
        {
            // computed
            this.out_sx = def.in_sx;
            this.out_sy = def.in_sy;
            this.out_depth = def.in_depth;
            this.type = "dropout";
            this.drop_prob = def.drop_prob != double.NaN ? def.drop_prob : 0.5;
            this.dropped = new bool[this.out_sx * this.out_sy * this.out_depth];
        }

        public override Volume forward(Volume V, bool is_training)
        {
            this.in_act = V;

            var V2 = V.clone();
            var N = V.w.Length;
            if (is_training)
            {
                // do dropout
                for (var i = 0; i < N; i++)
                {
                    if (util.random.NextDouble() < this.drop_prob)
                    {
                        // drop!
                        V2.w[i] = 0;
                        this.dropped[i] = true;
                    } 

                    else
                    {
                        this.dropped[i] = false;
                    }
                }
            }
            else
            {
                // scale the activations during prediction
                for (var i = 0; i < N; i++) { V2.w[i] *= this.drop_prob; }
            }
            this.out_act = V2;
            return this.out_act; // dummy identity function for now
        }

        public override double backward(object y)
        {
            var V = this.in_act; // we need to set dw of this
            var chain_grad = this.out_act;
            var N = V.w.Length;
            V.dw = util.zeros(N); // zero out gradient wrt data
            for (var i = 0; i < N; i++)
            {
                if (!(this.dropped[i]))
                {
                    V.dw[i] = chain_grad.dw[i]; // copy over the gradient
                }
            }

            return 0.0;
        }
        public override Gradient[] getParamsAndGrads()
        {
            return new List<Gradient>().ToArray();
        }
    }
}
