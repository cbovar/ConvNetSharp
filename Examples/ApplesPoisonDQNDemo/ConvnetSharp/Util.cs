using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvnetSharpOLD
{
    // a window stores _size_ number of values
    // and returns averages. Useful for keeping running
    // track of validation or training accuracy during SGD
    [Serializable]
    public class TrainingWindow
    {
        public List<double> v;
        public int size;
        public int minsize;
        public double sum;

        public TrainingWindow(int size, int minsize)
        {
            this.v = new List<double>();
            this.size = size <= minsize ? 100 : size;
            this.minsize = minsize <= 2 ? 20 : minsize;
            this.sum = 0;
        }

        public void add(double x)
        {
            this.v.Add(x);
            this.sum += x;
            if (this.v.Count > this.size)
            {
                var xold = this.v[0];
                v.RemoveAt(0);
                this.sum -= xold;
            }
        }

        public double get_average()
        {
            if (this.v.Count < this.minsize)
                return -1;
            else
                return this.sum / this.v.Count;
        }

        public void reset()
        {
            this.v = new List<double>();
            this.sum = 0;
        }
    }

    [Serializable]
    public class Util
    {
        // Random number utilities
        bool return_v = false;
        double v_val = 0.0;
        public Random random = new Random();

        public double gaussRandom()
        {
            if(return_v) { 
              return_v = false;
              return v_val; 
            }

            var u = 2 * random.NextDouble() - 1;
            var v = 2 * random.NextDouble() - 1;
            var r = u * u + v * v;
            if(r == 0 || r > 1) return gaussRandom();
            var c = Math.Sqrt(-2 * Math.Log(r) / r);
            v_val = v* c; // cache this
            return_v = true;
            return u* c;
        }

        public double Randf(double a, double b) { return random.NextDouble() * (b-a)+a; }
        public int RandInt(int a, int b) { return random.Next(a, b); }
        public double randn(double mu, double std) { return mu+gaussRandom()* std; }

        // Array utilities
        public double[] zeros(int n)
        {
            if (n <= 0)
            {
                return new double[] { 0.0 };
            }
            else
            {
                var arr = new double[n];
                for (var i = 0; i < n; i++) { arr[i] = 0; }
                return arr;
            }
        }

        public bool arrContains(object[] arr, object elt)
        {
            for (int i = 0, n = arr.Length; i < n; i++)
            {
                if (arr[i] == elt)
                    return true;
            }

            return false;
        }

        public object[] arrUnique(object[] arr)
        {
            var b = new List<object>();
            for (int i = 0, n = arr.Length; i < n; i++)
            {
                if (!arrContains(b.ToArray(), arr[i]))
                {
                    b.Add(arr[i]);
                }
            }
            return b.ToArray();
        }

        // sample from list lst according to probabilities in list probs
        // the two lists are of same size, and probs adds up to 1
        public double weightedSample(double[] lst, double[] probs) {
            double p = Randf(0, 1.0);
            var cumprob = 0.0;

            for (int k = 0, n = lst.Length; k < n; k++) {
                cumprob += probs[k];
                if (p < cumprob) { return lst[k]; }
            }

            return p;
        }

        // syntactic sugar function for getting default parameter values
        public string getopt(string opt_obj, object field_name, string default_value) {

            var opt = JsonConvert.DeserializeObject<Dictionary<string, string>>(opt_obj);

            if (field_name.GetType().Equals(typeof(string))) {

                // case of single string
                if (opt.ContainsKey((string)field_name))
                {
                    return (string.IsNullOrEmpty(opt[(string)field_name])) ? opt[(string)field_name] : default_value;
                }
                else
                {
                    return default_value;
                }
            } else {
                // assume we are given an array of string instead
                var ret = default_value;
                var fields = (string[])field_name;

                for (var i = 0; i< fields.Length;i++) {
                    var field = fields[i];

                    if(opt.ContainsKey(field))
                        ret = opt[field];
                }

                return ret;
            }
        }

        public void assert(Boolean condition, string message)
        {
            if (!condition)
            {
                throw new Exception(message);
            }
        }
    }
}
