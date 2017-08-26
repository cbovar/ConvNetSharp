using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvnetSharpOLD
{
    // Volume is the basic building block of all data in a net.
    // it is essentially just a 3D volume of numbers, with a
    // width (sx), height (sy), and depth (depth).
    // it is used to hold data for all filters, all volumes,
    // all weights, and also stores all gradients w.r.t. 
    // the data. c is optionally a value to initialize the volume
    // with. If c is missing, fills the Vol with random numbers.
    [Serializable]
    public class Volume
    {
        public int sx, sy, depth;

        public double[] w;
        public double[] dw;

        Util util = new Util();

        public Volume(int sx, int sy, int depth)
        {
            Init(sx, sy, depth, double.MinValue);
        }

        public Volume(int sx, int sy, int depth, double c)
        {
            Init(sx, sy, depth, c);
        }

        private void Init(int sx, int sy, int depth, double c)
        {
            // we were given dimensions of the vol
            this.sx = sx;
            this.sy = sy;
            this.depth = depth;

            var n = sx * sy * depth;
            this.w = util.zeros(n);
            this.dw = util.zeros(n);

            if (c == double.MinValue)
            {
                // weight normalization is done to equalize the output
                // variance of every neuron, otherwise neurons with a lot
                // of incoming connections have outputs of larger variance
                var scale = Math.Sqrt(1.0 / (sx * sy * depth));
                for (var i = 0; i < n; i++)
                {
                    this.w[i] = util.randn(0.0, scale);
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    this.w[i] = c;
                }
            }
        }

        public double get(int x, int y, int d)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            return this.w[ix];
        }

        public void set(int x, int y, int d, double v)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            this.w[ix] = v;
        }

        public void add(int x, int y, int d, double v)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            this.w[ix] += v;
        }

        public double get_grad(int x, int y, int d)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            return this.dw[ix];
        }

        public void set_grad(int x, int y, int d, double v)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            this.dw[ix] = v;
        }

        public void add_grad(int x, int y, int d, double v)
        {
            var ix = ((this.sx * y) + x) * this.depth + d;
            this.dw[ix] += v;
        }

        public Volume cloneAndZero() { return new Volume(this.sx, this.sy, this.depth, 0.0); }
        public Volume clone()
        {
            var V = new Volume(this.sx, this.sy, this.depth, 0.0);
            var n = this.w.Length;
            for (var i = 0; i < n; i++) { V.w[i] = this.w[i]; }
            return V;
        }

        public void addFrom(Volume V) { for (var k = 0; k < this.w.Length; k++) { this.w[k] += V.w[k]; } }
        public void addFromScaled(Volume V, double a) { for (var k = 0; k < this.w.Length; k++) { this.w[k] += a * V.w[k]; } }
        public void setConst(double a) { for (var k = 0; k < this.w.Length; k++) { this.w[k] = a; } }
    }
}
