using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvnetSharpOLD
{
    // - ConvLayer does convolutions (so weight sharing spatially)
    [Serializable]
    public class ConvLayer : LayerBase
    {
        Util util = new Util();

        Volume biases;
        int stride, pad;

        public ConvLayer(LayerDefinition def) : base()
        {
            // required
            this.out_depth = def.n_filters;
            this.sx = def.sx; // filter size. Should be odd if possible, it's cleaner.
            this.in_depth = def.in_depth;
            this.in_sx = def.in_sx;
            this.in_sy = def.in_sy;

            // optional
            this.sy = def.sy != int.MinValue ? def.sy : this.sx;
            this.stride = def.stride != int.MinValue ? def.stride : 1; // stride at which we apply filters to input volume
            this.pad = def.pad != int.MinValue ? def.pad : 0; // amount of 0 padding to add around borders of input volume
            this.l1_decay_mul = def.l1_decay_mul != double.MinValue ? def.l1_decay_mul : 0.0;
            this.l2_decay_mul = def.l2_decay_mul != double.MinValue ? def.l2_decay_mul : 1.0;

            // computed
            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            this.out_sx = (int)Math.Floor((double)(def.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
            this.out_sy = (int)Math.Floor((double)(def.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
            this.type = "conv";

            // initializations
            var bias = def.bias_pref != double.MinValue ? def.bias_pref : 0.0;
            this.filters = new List<Volume>();
            for (var i = 0; i < this.out_depth; i++) { this.filters.Add(new Volume(this.sx, this.sy, this.in_depth)); }
            this.biases = new Volume(1, 1, this.out_depth, bias);
        }

        public override Volume forward(Volume V, bool is_training)
        {
            // optimized code by @mdda that achieves 2x speedup over previous version

            this.in_act = V;
            var A = new Volume(this.out_sx | 0, this.out_sy | 0, this.out_depth | 0, 0.0);

            var V_sx = V.sx | 0;
            var V_sy = V.sy | 0;
            var xy_stride = this.stride | 0;

            for (var d = 0; d < this.out_depth; d++)
            {
                var f = this.filters[d];
                var x = -this.pad | 0;
                var y = -this.pad | 0;
                for (var ay = 0; ay < this.out_sy; y += xy_stride, ay++)
                {  // xy_stride
                    x = -this.pad | 0;
                    for (var ax = 0; ax < this.out_sx; x += xy_stride, ax++)
                    {  // xy_stride

                        // convolve centered at this particular location
                        var a = 0.0;
                        for (var fy = 0; fy < f.sy; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < f.sx; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx)
                                {
                                    for (var fd = 0; fd < f.depth; fd++)
                                    {
                                        // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                        a += f.w[((f.sx * fy) + fx) * f.depth + fd] * V.w[((V_sx * oy) + ox) * V.depth + fd];
                                    }
                                }
                            }
                        }
                        a += this.biases.w[d];
                        A.set(ax, ay, d, a);
                    }
                }
            }
            this.out_act = A;
            return this.out_act;
        }

        public override double backward(object _y)
        {
            var V = this.in_act;
            V.dw = util.zeros(V.w.Length); // zero out gradient wrt bottom data, we're about to fill it

            var V_sx = V.sx | 0;
            var V_sy = V.sy | 0;
            var xy_stride = this.stride | 0;

            for (var d = 0; d < this.out_depth; d++)
            {
                var f = this.filters[d];
                var x = -this.pad | 0;
                var y = -this.pad | 0;
                for (var ay = 0; ay < this.out_sy; y += xy_stride, ay++)
                {  // xy_stride
                    x = -this.pad | 0;
                    for (var ax = 0; ax < this.out_sx; x += xy_stride, ax++)
                    {  // xy_stride

                        // convolve centered at this particular location
                        var chain_grad = this.out_act.get_grad(ax, ay, d); // gradient from above, from chain rule
                        for (var fy = 0; fy < f.sy; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < f.sx; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx)
                                {
                                    for (var fd = 0; fd < f.depth; fd++)
                                    {
                                        // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                        var ix1 = ((V_sx * oy) + ox) * V.depth + fd;
                                        var ix2 = ((f.sx * fy) + fx) * f.depth + fd;
                                        f.dw[ix2] += V.w[ix1] * chain_grad;
                                        V.dw[ix1] += f.w[ix2] * chain_grad;
                                    }
                                }
                            }
                        }
                        this.biases.dw[d] += chain_grad;
                    }
                }
            }

            return 0.0;
        }
        public override Gradient[] getParamsAndGrads()
        {
            var response = new List<Gradient>();
            for (var i = 0; i < this.out_depth; i++)
            {
                response.Add(new Gradient { w = this.filters[i].w, dw = this.filters[i].dw, l2_decay_mul = this.l2_decay_mul, l1_decay_mul = this.l1_decay_mul });
            }
            response.Add(new Gradient { w = this.biases.w, dw = this.biases.dw, l1_decay_mul = 0.0, l2_decay_mul = 0.0 });
            return response.ToArray();
        }
    }
}
