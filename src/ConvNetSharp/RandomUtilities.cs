using System;

namespace ConvNetSharp
{
    public static class RandomUtilities
    {
        public static Random Random = new Random((int) DateTime.Now.Ticks);
        private static double v_val;
        private static bool return_v;

        public static double GaussianRandom()
        {
            if (return_v)
            {
                return_v = false;
                return v_val;
            }
            var u = 2*Random.NextDouble() - 1;
            var v = 2*Random.NextDouble() - 1;
            var r = u*u + v*v;
            if (r == 0 || r > 1)
            {
                return GaussianRandom();
            }

            var c = Math.Sqrt(-2*Math.Log(r)/r);
            v_val = v*c; // cache this
            return_v = true;
            return u*c;
        }

        public static double Randn(double mu, double std)
        {
            return mu + GaussianRandom()*std;
        }
    }
}