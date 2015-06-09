using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace ConvNetSharp
{
    public enum Transformation
    {
        Convolution,
        DynamicTimeWarp
    }


    public class DynamicTimeWarpLayer : LayerBase
    {
        private Volume biases;

        public DynamicTimeWarpLayer(int width, int filterCount)
        {
            this.GroupSize = 2;
            this.L1DecayMul = 0.0;
            this.L2DecayMul = 1.0;
            this.Stride = 1;
            this.Pad = 0;

            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = 1;
        }

        public List<Volume> Filters { get; private set; }

        public int FilterCount { get; private set; }

        public double L1DecayMul { get; set; }

        public double L2DecayMul { get; set; }

        public int Stride { get; set; }

        public int Pad { get; set; }

        public double BiasPref { get; set; }

        public Activation Activation { get; set; }

        public int GroupSize { get; private set; }

        public Transformation Transformation { get; set; }

        public override Volume Forward(Volume volume, bool isTraining = false)
        {
            this.InputActivation = volume;
            var outputActivation = new Volume(this.OutputWidth, 1, this.OutputDepth, 0.0);

            for (var depth = 0; depth < this.OutputDepth; depth++)
            {
                var filter = this.Filters[depth];
                var x = -this.Pad;
                for (var ax = 0; ax < this.OutputWidth; x += this.Stride, ax++)
                {
                    var a = DTW(volume, filter, x);
                    //a += this.biases.Weights[depth];
                    outputActivation.Set(ax, 0, depth, a);
                }
            }

            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        private static double Convolve(Volume volume, Volume filter, int x)
        {
            // convolve centered at this particular location
            var a = 0.0;
            for (var fx = 0; fx < filter.Width; fx++)
            {
                var ox = x + fx;
                if (ox >= 0 && ox < volume.Width)
                {
                    a += filter.Weights[fx] * volume.Weights[ox];
                }
            }
            return a;
        }

        private static double Simple(Volume volume, Volume filter, int x)
        {
            // convolve centered at this particular location
            var a = 0.0;
            for (var fx = 0; fx < filter.Width; fx++)
            {
                var ox = x + fx;
                if (ox >= 0 && ox < volume.Width)
                {
                    a += Math.Abs(filter.Weights[fx] - volume.Weights[ox]);
                }
            }
            return a;
        }

        private static double DTW(Volume volume, Volume filter, int x)
        {
            // convolve centered at this particular location
            var a = 0.0;

            var t = new double[filter.Width];
            for (int fx = 0; fx < filter.Width; fx++)
            {
                var ox = x + fx;
                if (ox >= 0 && ox < volume.Width)
                {
                    t[fx] = volume.Weights[ox];
                }
            }

            a = Dtw(filter.Weights, t, 3);

            return a;
        }

        public static double Dtw(double[] s, double[] t, int w)
        {
            int n = s.Length;
            int m = t.Length;
            w = Math.Max(w, Math.Abs(n - m));
            var dtw = new double[n, m];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    dtw[i, j] = double.MaxValue;
                }
            }
            dtw[0, 0] = 0;

            for (int i = 1; i < n; i++)
            {
                var from = Math.Max(1, i - w);
                var to = Math.Min(m - 1, i + w);

                for (int j = from; j <= to; j++)
                {
                    var cost = Math.Sqrt((s[i] - t[j]) * (s[i] - t[j])); // distance
                    dtw[i, j] = cost + Math.Min(Math.Min(dtw[i - 1, j], dtw[i, j - 1]), dtw[i - 1, j - 1]);
                }
            }

            return dtw[n - 1, m - 1];
        }

        public override void Backward()
        {
            var volume = this.InputActivation;
            volume.WeightGradients = new double[volume.Weights.Length]; // zero out gradient wrt bottom data, we're about to fill it

            for (var depth = 0; depth < this.OutputDepth; depth++)
            {
                var filter = this.Filters[depth];

                // xyStride
                var x = -this.Pad;
                for (var ax = 0; ax < this.OutputWidth; x += this.Stride, ax++)
                {
                    // convolve centered at this particular location
                    var chainGradient = this.OutputActivation.GetGradient(ax, 0, depth);
                    // gradient from above, from chain rule
                    for (var fx = 0; fx < filter.Width; fx++)
                    {
                        var ox = x + fx;
                        if (ox >= 0 && ox < volume.Width)
                        {
                            // avoid function call overhead (x2) for efficiency, compromise modularity :(
                            filter.WeightGradients[fx] += chainGradient;
                            volume.WeightGradients[ox] += -chainGradient; ;
                            //filter.WeightGradients[fx] += (volume.Weights[ox] - filter.Weights[fx]) / Math.Abs(filter.Weights[fx] - volume.Weights[ox]) * chainGradient;
                            //volume.WeightGradients[ox] += -(volume.Weights[ox] - filter.Weights[fx]) / Math.Abs(volume.Weights[ox] - filter.Weights[fx]) * chainGradient; ;

                            //filter.WeightGradients[fx] += volume.Weights[ox] * chainGradient;
                            //volume.WeightGradients[ox] += filter.Weights[fx] * chainGradient;
                        }
                    }

                    this.biases.WeightGradients[depth] += chainGradient;
                }
            }
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            if (inputHeight != 1)
            {
                throw new ArgumentException("inputHeight must be 1", "inputHeight");
            }

            if (inputDepth != 1)
            {
                throw new ArgumentException("inputDepth must be 1", "inputDepth");
            }

            base.Init(inputWidth, inputHeight, inputDepth);

            // required
            this.OutputDepth = this.FilterCount;

            // computed
            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            this.OutputWidth = (int)Math.Floor((this.InputWidth + this.Pad * 2 - this.Width) / (double)this.Stride + 1);
            this.OutputHeight = 1;

            // initializations
            var bias = this.BiasPref;
            this.Filters = new List<Volume>();

            for (var i = 0; i < this.OutputDepth; i++)
            {
                this.Filters.Add(new Volume(this.Width, 1, 1));
            }

            this.biases = new Volume(1, 1, this.OutputDepth, bias);
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();
            for (var i = 0; i < this.OutputDepth; i++)
            {
                response.Add(new ParametersAndGradients
                {
                    Parameters = this.Filters[i].Weights,
                    Gradients = this.Filters[i].WeightGradients,
                    L2DecayMul = this.L2DecayMul,
                    L1DecayMul = this.L1DecayMul
                });
            }

            response.Add(new ParametersAndGradients
            {
                Parameters = this.biases.Weights,
                Gradients = this.biases.WeightGradients,
                L1DecayMul = 0.0,
                L2DecayMul = 0.0
            });

            return response;
        }
    }
}