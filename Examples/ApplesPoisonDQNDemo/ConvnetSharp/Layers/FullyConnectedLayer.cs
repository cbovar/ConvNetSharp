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
    public class FullyConnectedLayer : LayerBase
    {
        Volume biases;

        Util util = new Util();

        public FullyConnectedLayer(LayerDefinition def) : base()
        {
            // required
            this.out_depth = def.num_neurons;

            // optional 
            this.l1_decay_mul = def.l1_decay_mul != double.MinValue ? def.l1_decay_mul : 0.0;
            this.l2_decay_mul = def.l2_decay_mul != double.MinValue ? def.l2_decay_mul : 1.0;

            // computed
            this.num_inputs = def.in_sx * def.in_sy * def.in_depth;
            this.out_sx = 1;
            this.out_sy = 1;
            this.type = "fc";

            // initializations
            var bias = def.bias_pref != double.MinValue ? def.bias_pref : 0.0;
            this.filters = new List<Volume>();
            for (var i = 0; i < this.out_depth; i++) { this.filters.Add(new Volume(1, 1, this.num_inputs)); }
            this.biases = new Volume(1, 1, this.out_depth, bias);
        }

        public override Volume forward(Volume V, bool is_training)
        {
            this.in_act = V;
            var A = new Volume(1, 1, this.out_depth, 0.0);
            var Vw = V.w;
            for (var i = 0; i < this.out_depth; i++)
            {
                var a = 0.0;
                var wi = this.filters[i].w;
                for (var d = 0; d < this.num_inputs; d++)
                {
                    a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
                }
                a += this.biases.w[i];
                A.w[i] = a;
            }
            this.out_act = A;
            return this.out_act;
        }

        public override double backward(object y)
        {
            var V = this.in_act;
            V.dw = util.zeros(V.w.Length); // zero out the gradient in input Vol

            // compute gradient wrt weights and data
            for (var i = 0; i < this.out_depth; i++)
            {
                var tfi = this.filters[i];
                var chain_grad = this.out_act.dw[i];
                for (var d = 0; d < this.num_inputs; d++)
                {
                    V.dw[d] += tfi.w[d] * chain_grad; // grad wrt input data
                    tfi.dw[d] += V.w[d] * chain_grad; // grad wrt params
                }
                this.biases.dw[i] += chain_grad;
            }

            return 0.0;
        }
        public override Gradient[] getParamsAndGrads()
        {
            var response = new List<Gradient>();
            for (var i = 0; i < this.out_depth; i++)
            {
                response.Add(new Gradient { w=this.filters[i].w, dw=this.filters[i].dw, l1_decay_mul=this.l1_decay_mul, l2_decay_mul=this.l2_decay_mul});
            }

            response.Add(new Gradient { w=this.biases.w, dw=this.biases.dw, l1_decay_mul=0.0, l2_decay_mul=0.0});
            return response.ToArray();
        }
    }
}
