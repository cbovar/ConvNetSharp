using System;
using System.Collections;
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
    public class Volume : IVolume
    {
        [DataMember]
        private double[] WeightGradients;
        [DataMember]
        private double[] Weights;

        [DataMember]
        public int Width { get; private set; }

        [DataMember]
        public int Height { get; private set; }

        [DataMember]
        public int Depth { get; private set; }

        public int Length { get { return this.Weights.Length; } }

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
            var scale = Math.Sqrt(2.0 / (width * height * depth));

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

        public void AddGradient(int i, double v)
        {
            this.WeightGradients[i] += v;
        }

        public IVolume CloneAndZero()
        {
            return new Volume(this.Width, this.Height, this.Depth, 0.0);
        }

        public IVolume Clone()
        {
            var volume = new Volume(this.Width, this.Height, this.Depth, 0.0);
            var n = this.Weights.Length;

            for (var i = 0; i < n; i++)
            {
                volume.Weights[i] = this.Weights[i];
            }

            return volume;
        }

        public void ZeroGradients()
        {
            Array.Clear(this.WeightGradients,0, this.WeightGradients.Length);
        }

        public void AddFrom(IVolume volume)
        {
            for (var i = 0; i < this.Weights.Length; i++)
            {
                this.Weights[i] += volume.Get(i);
            }
        }

        public void AddGradientFrom(IVolume volume)
        {
            for (var i = 0; i < this.WeightGradients.Length; i++)
            {
                this.WeightGradients[i] += volume.GetGradient(i);
            }
        }

        public void AddFromScaled(IVolume volume, double a)
        {
            for (var i = 0; i < this.Weights.Length; i++)
            {
                this.Weights[i] += a * volume.Get(i);
            }
        }

        public void SetConst(double c)
        {
            for (var i = 0; i < this.Weights.Length; i++)
            {
                this.Weights[i] += c;
            }
        }

        public double Get(int i)
        {
            return this.Weights[i];
        }

        public double GetGradient(int i)
        {
            return this.WeightGradients[i];
        }

        public void SetGradient(int i, double v)
        {
            this.WeightGradients[i] = v; ;
        }

        public void Set(int i, double v)
        {
            this.Weights[i] = v;
        }

        public IEnumerator<double> GetEnumerator()
        {
            for(int i = 0; i < this.Length; i++)
            {
                yield return this.Weights[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
    }
}