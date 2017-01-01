using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

namespace ConvNetSharp
{
    /// <summary>
    ///     Volume is the basic building block of all data in a net.
    ///     it is essentially just a 3D volume of numbers, with a
    ///     width, height, and depth.
    ///     it is used to hold data for all filters, all volumes,
    ///     all weights, and also stores all gradients w.r.t.
    ///     the data.
    /// </summary>
    [DataContract]
    [Serializable]
    public class Volume
    {
        [DataMember]
        public int Depth;
        [DataMember]
        public int Height;
        [DataMember]
        public double[] WeightGradients;
        [DataMember]
        public double[] Weights;
        [DataMember]
        public int Width;

        /// <summary>
        ///     Volume will be filled with random numbers
        /// </summary>
        /// <param name="width">width</param>
        /// <param name="height">height</param>
        /// <param name="depth">depth</param>
        public Volume(int width, int height, int depth)
        {
            // we were given dimensions of the vol
            this.Width = width;
            this.Height = height;
            this.Depth = depth;

            var n = width * height * depth;
            this.Weights = new double[n];
            this.WeightGradients = new double[n];

            // weight normalization is done to equalize the output
            // variance of every neuron, otherwise neurons with a lot
            // of incoming connections have outputs of larger variance
            var scale = Math.Sqrt(1.0 / (width * height * depth));

            for (var i = 0; i < n; i++)
            {
                this.Weights[i] = RandomUtilities.Randn(0.0, scale);
            }
        }

        /// <summary>
        /// </summary>
        /// <param name="width">width</param>
        /// <param name="height">height</param>
        /// <param name="depth">depth</param>
        /// <param name="c">value to initialize the volume with</param>
        public Volume(int width, int height, int depth, double c)
        {
            // we were given dimensions of the vol
            this.Width = width;
            this.Height = height;
            this.Depth = depth;

            var n = width * height * depth;
            this.Weights = new double[n];
            this.WeightGradients = new double[n];

            if (c != 0)
            {
                for (var i = 0; i < n; i++)
                {
                    this.Weights[i] = c;
                }
            }
        }

        public Volume(IList<double> weights)
        {
            // we were given a list in weights, assume 1D volume and fill it up
            this.Width = 1;
            this.Height = 1;
            this.Depth = weights.Count;

            this.Weights = new double[this.Depth];
            this.WeightGradients = new double[this.Depth];

            for (var i = 0; i < this.Depth; i++)
            {
                this.Weights[i] = weights[i];
            }
        }

        public double Get(int x, int y, int d)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            return this.Weights[ix];
        }

        public void Set(int x, int y, int d, double v)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            this.Weights[ix] = v;
        }

        public void Add(int x, int y, int d, double v)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            this.Weights[ix] += v;
        }

        public double GetGradient(int x, int y, int d)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            return this.WeightGradients[ix];
        }

        public void SetGradient(int x, int y, int d, double v)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            this.WeightGradients[ix] = v;
        }

        public void AddGradient(int x, int y, int d, double v)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            this.WeightGradients[ix] += v;
        }

        public Volume CloneAndZero()
        {
            return new Volume(this.Width, this.Height, this.Depth, 0.0);
        }

        public Volume Clone()
        {
            var volume = new Volume(this.Width, this.Height, this.Depth, 0.0);
            var n = this.Weights.Length;

            for (var i = 0; i < n; i++)
            {
                volume.Weights[i] = this.Weights[i];
            }

            return volume;
        }

        public void AddFrom(Volume volume)
        {
            for (var i = 0; i < this.Weights.Length; i++)
            {
                this.Weights[i] += volume.Weights[i];
            }
        }

        public void AddGradientFrom(Volume volume)
        {
            for (var i = 0; i < this.WeightGradients.Length; i++)
            {
                this.WeightGradients[i] += volume.WeightGradients[i];
            }
        }

        public void AddFromScaled(Volume volume, double a)
        {
            for (var i = 0; i < this.Weights.Length; i++)
            {
                this.Weights[i] += a * volume.Weights[i];
            }
        }

        public void SetConst(double c)
        {
            for (var i = 0; i < this.Weights.Length; i++)
            {
                this.Weights[i] += c;
            }
        }
    }
}