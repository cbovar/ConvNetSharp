using System;

namespace ConvNetSharp
{
    public static class RandomUtilities
    {
        private static readonly Random Random = new Random(Seed);
        private static double val;
        private static bool returnVal;

        public static int Seed
        {
            get { return (int) DateTime.Now.Ticks; }
        }

        public static double GaussianRandom()
        {
            if (returnVal)
            {
                returnVal = false;
                return val;
            }

            var u = 2*Random.NextDouble() - 1;
            var v = 2*Random.NextDouble() - 1;
            var r = u*u + v*v;

            if (r == 0 || r > 1)
            {
                return GaussianRandom();
            }

            var c = Math.Sqrt(-2*Math.Log(r)/r);
            val = v*c; // cache this
            returnVal = true;

            return u*c;
        }

        public static double Randn(double mu, double std)
        {
            return mu + GaussianRandom()*std;
        }
    }
}