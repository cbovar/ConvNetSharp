using System;

namespace ConvNetSharp.Volume
{
    public static class RandomUtilities
    {
        private static readonly Random Random = new Random(Seed);
        private static double val;
        private static bool returnVal;

        public static int Seed => (int) 123;//DateTime.Now.Ticks;

        public static double GaussianRandom()
        {
            if (returnVal)
            {
                returnVal = false;
                return val;
            }

            double r = 0, u = 0, v = 0;

            //System.Random is not threadsafe
            lock (Random)
            {
                while (r == 0 || r > 1)
                {
                    u = 2*Random.NextDouble() - 1;
                    v = 2*Random.NextDouble() - 1;
                    r = u*u + v*v;
                }
            }

            var c = Math.Sqrt(-2 * Math.Log(r) / r);
            val = v * c; // cache this
            returnVal = true;

            return u * c;
        }

        public static double NextDouble()
        {
            return Random.NextDouble();
        }

        public static double Randn(double mu, double std)
        {
            return mu + GaussianRandom() * std;
        }

        public static double[] RandomDoubleArray(long length, double mu = 0.0, double std = 1.0, bool posisitveOnly = false)
        {
            var values = new double[length];

            for (var i = 0; i < length; i++)
            {
                values[i] = Randn(mu, std);
                if (posisitveOnly)
                {
                    values[i] = Math.Abs(values[i]);
                }
            }

            return values;
        }

        public static float[] RandomSingleArray(long length, double mu = 0.0, double std = 1.0, bool posisitveOnly = false)
        {
            var values = new float[length];

            for (var i = 0; i < length; i++)
            {
                values[i] = (float) Randn(mu, std);
                if (posisitveOnly)
                {
                    values[i] = Math.Abs(values[i]);
                }
            }

            return values;
        }
    }
}