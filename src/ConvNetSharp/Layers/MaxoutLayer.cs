using System;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    /// <summary>
    ///     Implements Maxout nnonlinearity that computes
    ///     x -> max(x)
    ///     where x is a vector of size group_size. Ideally of course,
    ///     the input size should be exactly divisible by group_size
    /// </summary>
    [DataContract]
    [Serializable]
    public class MaxoutLayer : LayerBase
    {
        [DataMember]
        private int[] switches;

        public MaxoutLayer(int groupSize = 2)
        {
            this.GroupSize = groupSize;
        }

        [DataMember]
        public int GroupSize { get; set; }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var depth = this.OutputDepth;
            var outputActivation = new Volume(this.OutputWidth, this.OutputHeight, this.OutputDepth, 0.0);

            // optimization branch. If we're operating on 1D arrays we dont have
            // to worry about keeping track of x,y,d coordinates inside
            // input volumes. In convnets we do :(
            if (this.OutputWidth == 1 && this.OutputHeight == 1)
            {
                for (var i = 0; i < depth; i++)
                {
                    var ix = i * this.GroupSize; // base index offset
                    var a = input.Get(ix);
                    var ai = 0;

                    for (var j = 1; j < this.GroupSize; j++)
                    {
                        var a2 = input.Get(ix + j);
                        if (a2 > a)
                        {
                            a = a2;
                            ai = j;
                        }
                    }

                    outputActivation.Set(i, a);
                    this.switches[i] = ix + ai;
                }
            }
            else
            {
                var n = 0; // counter for switches
                for (var x = 0; x < input.Width; x++)
                {
                    for (var y = 0; y < input.Height; y++)
                    {
                        for (var i = 0; i < depth; i++)
                        {
                            var ix = i * this.GroupSize;
                            var a = input.Get(x, y, ix);
                            var ai = 0;

                            for (var j = 1; j < this.GroupSize; j++)
                            {
                                var a2 = input.Get(x, y, ix + j);
                                if (a2 > a)
                                {
                                    a = a2;
                                    ai = j;
                                }
                            }

                            outputActivation.Set(x, y, i, a);
                            this.switches[n] = ix + ai;
                            n++;
                        }
                    }
                }
            }

            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation; // we need to set dw of this
            var volume2 = this.OutputActivation;
            var depth = this.OutputDepth;
            volume.ZeroGradients(); // zero out gradient wrt data

            // pass the gradient through the appropriate switch
            if (this.OutputWidth == 1 && this.OutputHeight == 1)
            {
                for (var i = 0; i < depth; i++)
                {
                    var chainGradient = volume2.GetGradient(i);
                    volume.SetGradient(this.switches[i], chainGradient);
                }
            }
            else
            {
                // bleh okay, lets do this the hard way
                var n = 0; // counter for switches
                for (var x = 0; x < volume2.Width; x++)
                {
                    for (var y = 0; y < volume2.Height; y++)
                    {
                        for (var i = 0; i < depth; i++)
                        {
                            var chainGradient = volume2.GetGradient(x, y, i);
                            volume.SetGradient(x, y, this.switches[n], chainGradient);
                            n++;
                        }
                    }
                }
            }
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.OutputDepth = (int)Math.Floor(inputDepth / (double)this.GroupSize);
            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;

            this.switches = new int[this.OutputWidth * this.OutputHeight * this.OutputDepth]; // useful for backprop
        }
    }
}