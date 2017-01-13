using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace ConvNetSharp.Layers
{
    [DataContract]
    [Serializable]
    public class ConvLayer : LayerBase, IDotProductLayer
    {
        private int stride = 1;
        private int pad;

        public ConvLayer(int width, int height, int filterCount)
        {
            this.L1DecayMul = 0.0;
            this.L2DecayMul = 1.0;

            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = height;
        }

        [DataMember]
        public int Width { get; private set; }

        [DataMember]
        public int Height { get; private set; }

        [DataMember]
        public Volume Biases { get; private set; }

        [DataMember]
        public List<Volume> Filters { get; private set; }

        [DataMember]
        public int FilterCount { get; private set; }

        [DataMember]
        public double L1DecayMul { get; set; }

        [DataMember]
        public double L2DecayMul { get; set; }

        [DataMember]
        public int Stride
        {
            get
            {
                return this.stride;
            }
            set
            {
                this.stride = value;
            }
        }

        [DataMember]
        public int Pad
        {
            get
            {
                return this.pad;
            }
            set
            {
                this.pad = value;
            }
        }

        [DataMember]
        public double BiasPref { get; set; }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            // optimized code by @mdda that achieves 2x speedup over previous version

            this.InputActivation = input;
            var outputActivation = new Volume(this.OutputWidth, this.OutputHeight, this.OutputDepth, 0.0);

            var volumeWidth = input.Width;
            var volumeHeight = input.Height;
            var xyStride = this.Stride;

#if PARALLEL
            Parallel.For(0, this.OutputDepth, depth =>
#else
            for (var depth = 0; depth < this.OutputDepth; depth++)
#endif
            {
                var filter = this.Filters[depth];
                var y = -this.Pad;

                for (var ay = 0; ay < this.OutputHeight; y += xyStride, ay++)
                {
                    // xyStride
                    var x = -this.Pad;
                    for (var ax = 0; ax < this.OutputWidth; x += xyStride, ax++)
                    {
                        // xyStride

                        // convolve centered at this particular location
                        var a = 0.0;
                        for (var fy = 0; fy < filter.Height; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < filter.Width; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < volumeHeight && ox >= 0 && ox < volumeWidth)
                                {
                                    for (var fd = 0; fd < filter.Depth; fd++)
                                    {
                                        // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                        a += filter.Get((filter.Width * fy + fx) * filter.Depth + fd) *
                                             input.Get((volumeWidth * oy + ox) * input.Depth + fd);
                                    }
                                }
                            }
                        }

                        a += this.Biases.Get(depth);
                        outputActivation.Set(ax, ay, depth, a);
                    }
                }
            }
#if PARALLEL
                );
#endif

            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation;
            volume.ZeroGradients(); // zero out gradient wrt bottom data, we're about to fill it

            var volumeWidth = volume.Width;
            var volumeHeight = volume.Height;
            var volumeDepth = volume.Depth;
            var xyStride = this.Stride;

#if PARALLEL
            var locker = new object();
            Parallel.For(0, this.OutputDepth, () => new Volume(volumeWidth, volumeHeight, volumeDepth, 0), (depth, state, temp) =>
#else
            var temp = volume;
            for (var depth = 0; depth < this.OutputDepth; depth++)
#endif
            {
                var filter = this.Filters[depth];
                var y = -this.Pad;
                for (var ay = 0; ay < this.OutputHeight; y += xyStride, ay++)
                {
                    // xyStride
                    var x = -this.Pad;
                    for (var ax = 0; ax < this.OutputWidth; x += xyStride, ax++)
                    {
                        // xyStride

                        // convolve centered at this particular location
                        var chainGradient = this.OutputActivation.GetGradient(ax, ay, depth);
                        // gradient from above, from chain rule
                        for (var fy = 0; fy < filter.Height; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < filter.Width; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < volumeHeight && ox >= 0 && ox < volumeWidth)
                                {
                                    for (var fd = 0; fd < filter.Depth; fd++)
                                    {
                                        filter.AddGradient(fx, fy, fd, volume.Get(ox, oy, fd) * chainGradient);
                                        temp.AddGradient(ox, oy, fd, filter.Get(fx, fy, fd) * chainGradient);
                                    }
                                }
                            }
                        }

                        this.Biases.SetGradient(depth, this.Biases.GetGradient(depth) + chainGradient);
                    }
                }

#if !PARALLEL
            }
#else
                return temp;
            }
                ,
                result =>
                {
                    lock (locker)
                    {
                        volume.AddGradientFrom(result);
                    }
                });
#endif
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.UpdateOutputSize();
        }

        internal void UpdateOutputSize()
        {
            // required
            this.OutputDepth = this.FilterCount;

            // computed
            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            this.OutputWidth = (int)Math.Floor((this.InputWidth + this.Pad * 2 - this.Width) / (double)this.Stride + 1);
            this.OutputHeight = (int)Math.Floor((this.InputHeight + this.Pad * 2 - this.Height) / (double)this.Stride + 1);

            // initializations
            var bias = this.BiasPref;
            this.Filters = new List<Volume>();

            for (var i = 0; i < this.OutputDepth; i++)
            {
                this.Filters.Add(new Volume(this.Width, this.Height, this.InputDepth));
            }

            this.Biases = new Volume(1, 1, this.OutputDepth, bias);
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();
            for (var i = 0; i < this.OutputDepth; i++)
            {
                response.Add(new ParametersAndGradients
                {
                    //Parameters = this.Filters[i].Weights,
                    //Gradients = this.Filters[i].WeightGradients,
                    Volume = this.Filters[i],
                    L2DecayMul = this.L2DecayMul,
                    L1DecayMul = this.L1DecayMul
                });
            }

            response.Add(new ParametersAndGradients
            {
                //Parameters = this.Biases.Weights,
                //Gradients = this.Biases.WeightGradients,
                Volume = this.Biases,
                L1DecayMul = 0.0,
                L2DecayMul = 0.0
            });

            return response;
        }
    }
}