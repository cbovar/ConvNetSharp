using System;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace ConvNetSharp.Layers
{
    [DataContract]
    [Serializable]
    public class PoolLayer : LayerBase
    {
        private int stride = 2;
        private int pad;

        [DataMember]
        private int[] switchx;

        [DataMember]
        private int[] switchy;

        public PoolLayer(int width, int height)
        {
            this.Width = width;
            this.Height = height;
        }

        [DataMember(Order = 0)]
        public int Width { get; private set; }

        [DataMember(Order = 0)]
        public int Height { get; private set; }

        [DataMember(Order = 1)]
        public int Stride
        {
            get
            {
                return this.stride;
            }
            set
            {
                this.stride = value;
                this.UpdateOutputSize();
            }
        }

        [DataMember(Order = 1)]
        public int Pad
        {
            get
            {
                return this.pad;
            }
            set
            {
                this.pad = value;
                this.UpdateOutputSize();
            }
        }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;

            var outputActivation = new Volume(this.OutputWidth, this.OutputHeight, this.OutputDepth, 0.0);

#if PARALLEL
            Parallel.For(0, this.OutputDepth, depth =>
#else
            for (var depth = 0; depth < this.OutputDepth; depth++)
#endif
            {
                var n = depth * this.OutputWidth * this.OutputHeight; // a counter for switches

                var x = -this.Pad;
                for (var ax = 0; ax < this.OutputWidth; x += this.Stride, ax++)
                {
                    var y = -this.Pad;
                    for (var ay = 0; ay < this.OutputHeight; y += this.Stride, ay++)
                    {
                        // convolve centered at this particular location
                        var a = double.MinValue;
                        int winx = -1, winy = -1;

                        for (var fx = 0; fx < this.Width; fx++)
                        {
                            for (var fy = 0; fy < this.Height; fy++)
                            {
                                var oy = y + fy;
                                var ox = x + fx;
                                if (oy >= 0 && oy < input.Height && ox >= 0 && ox < input.Width)
                                {
                                    var v = input.Get(ox, oy, depth);
                                    // perform max pooling and store pointers to where
                                    // the max came from. This will speed up backprop 
                                    // and can help make nice visualizations in future
                                    if (v > a)
                                    {
                                        a = v;
                                        winx = ox;
                                        winy = oy;
                                    }
                                }
                            }
                        }

                        this.switchx[n] = winx;
                        this.switchy[n] = winy;
                        n++;
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
            // pooling layers have no parameters, so simply compute 
            // gradient wrt data here
            var volume = this.InputActivation;
            volume.ZeroGradients(); // zero out gradient wrt data

#if PARALLEL
            Parallel.For(0, this.OutputDepth, depth =>
#else
            for (var depth = 0; depth < this.OutputDepth; depth++)
#endif
            {
                var n = depth * this.OutputWidth * this.OutputHeight;

                for (var ax = 0; ax < this.OutputWidth; ax++)
                {
                    for (var ay = 0; ay < this.OutputHeight; ay++)
                    {
                        var chainGradient = this.OutputActivation.GetGradient(ax, ay, depth);
                        volume.AddGradient(this.switchx[n], this.switchy[n], depth, chainGradient);
                        n++;
                    }
                }
            }
#if PARALLEL
                );
#endif
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.UpdateOutputSize();
        }

        private void UpdateOutputSize()
        {
            // computed
            this.OutputDepth = this.InputDepth;
            this.OutputWidth = (int)Math.Floor((this.InputWidth + this.Pad * 2 - this.Width) / (double)this.Stride + 1);
            this.OutputHeight = (int)Math.Floor((this.InputHeight + this.Pad * 2 - this.Height) / (double)this.Stride + 1);

            // store switches for x,y coordinates for where the max comes from, for each output neuron
            this.switchx = new int[this.OutputWidth * this.OutputHeight * this.OutputDepth];
            this.switchy = new int[this.OutputWidth * this.OutputHeight * this.OutputDepth];
        }
    }
}