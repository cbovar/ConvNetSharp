using System;

namespace ConvNetSharp
{
    public class PoolLayer : LayerBase
    {
        private int[] switchx;
        private int[] switchy;

        public PoolLayer(int width, int height)
        {
            this.Width = width;
            this.Height = height;
            this.Stride = 2;
            this.Pad = 0;
        }

        public int Pad { get; set; }

        public int Stride { get; set; }

        public override Volume Forward(Volume volume, bool isTraining = false)
        {
            this.InputActivation = volume;

            var outputActivation = new Volume(this.OutputWidth, this.OutputHeight, this.OutputDepth, 0.0);

            var n = 0; // a counter for switches
            for (var depth = 0; depth < this.OutputDepth; depth++)
            {
                var x = -this.Pad;
                var y = -this.Pad;
                for (var ax = 0; ax < this.OutputWidth; x += this.Stride, ax++)
                {
                    y = -this.Pad;
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
                                if (oy >= 0 && oy < volume.Height && ox >= 0 && ox < volume.Width)
                                {
                                    var v = volume.Get(ox, oy, depth);
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

            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            // pooling layers have no parameters, so simply compute 
            // gradient wrt data here
            var volume = this.InputActivation;
            volume.WeightGradients = new double[volume.Weights.Length]; // zero out gradient wrt data

            var n = 0;
            for (var depth = 0; depth < this.OutputDepth; depth++)
            {
                var x = -this.Pad;
                var y = -this.Pad;
                for (var ax = 0; ax < this.OutputWidth; x += this.Stride, ax++)
                {
                    y = -this.Pad;
                    for (var ay = 0; ay < this.OutputHeight; y += this.Stride, ay++)
                    {
                        var chainGradient = this.OutputActivation.GetGradient(ax, ay, depth);
                        volume.AddGradient(this.switchx[n], this.switchy[n], depth, chainGradient);
                        n++;
                    }
                }
            }
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

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