using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace ConvNetSharp
{
    [DataContract]
    [Serializable]
    public class VolumeWrapper : IVolume
    {
        private int totalWeights;
        private IVolume[] volumes;
        private short[] indexes;
        private int[] starts;

        public VolumeWrapper(IEnumerable<IVolume> volumes)
        {
            this.volumes = volumes.ToArray();
            this.starts = new int[this.volumes.Length + 1];

            foreach (var vol in this.volumes)
            {
                this.totalWeights += vol.Length;
            }

            int i = 0;
            short k = 0;
            this.indexes = new short[totalWeights];
            this.starts[0] = 0;

            foreach (var vol in this.volumes)
            {
                for (int j = 0; j < vol.Length; j++)
                {
                    this.indexes[i++] = k;
                }

                k++;
                this.starts[k] = i;
            }

            this.Width = 1;
            this.Height = 1;
            this.Depth = totalWeights;
        }

        [DataMember]
        public int Width { get; private set; }

        [DataMember]
        public int Height { get; private set; }

        [DataMember]
        public int Depth { get; private set; }

        public int Length
        {
            get { return this.totalWeights; }
        }

        public void Add(int x, int y, int d, double v)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            var volIndex = this.indexes[ix];
            ix -= this.starts[volIndex];

            var val = this.volumes[volIndex].GetWeight(ix);
            this.volumes[volIndex].SetWeight(ix, v + val);
        }

        public void AddFrom(IVolume volume)
        {
            throw new NotImplementedException();
        }

        public void AddFromScaled(IVolume volume, double a)
        {
            throw new NotImplementedException();
        }

        public void AddGradient(int x, int y, int d, double v)
        {
            throw new NotImplementedException();
        }

        public void AddGradientFrom(IVolume volume)
        {
            throw new NotImplementedException();
        }

        public IVolume Clone()
        {
            var weights = new double[this.totalWeights];
            
            for (int i = 0; i < this.totalWeights; i++)
            {
                var volIndex = this.indexes[i];
                var index = i - this.starts[volIndex];

                weights[i] = this.volumes[volIndex].GetWeight(index);
            }

            return new Volume(weights);
        }

        public IVolume CloneAndZero()
        {
            throw new NotImplementedException();
        }

        public double Get(int x, int y, int d)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            var volIndex = this.indexes[ix];
            ix -= this.starts[volIndex];

            return this.volumes[volIndex].GetWeight(ix);
        }

        public double GetGradient(int x, int y, int d)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            var volIndex = this.indexes[ix];
            ix -= this.starts[volIndex];

            return this.volumes[volIndex].GetWeightGradient(ix);
        }

        public double GetWeight(int i)
        {
            var volIndex = this.indexes[i];
            i -= this.starts[volIndex];

            return this.volumes[volIndex].GetWeight(i);
        }

        public double GetWeightGradient(int i)
        {
            var volIndex = this.indexes[i];
            i -= this.starts[volIndex];

            return this.volumes[volIndex].GetWeightGradient(i);
        }

        public void Set(int x, int y, int d, double v)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            var volIndex = this.indexes[ix];
            ix -= this.starts[volIndex];

            this.volumes[volIndex].SetWeight(ix, v);
        }

        public void SetConst(double c)
        {
            foreach (var vol in this.volumes)
            {
                vol.SetConst(c);
            }
        }

        public void SetGradient(int x, int y, int d, double v)
        {
            var ix = ((this.Width * y) + x) * this.Depth + d;
            var volIndex = this.indexes[ix];
            ix -= this.starts[volIndex];

            this.volumes[volIndex].SetWeightGradient(ix, v);
        }

        public void SetWeight(int i, double v)
        {
            var volIndex = this.indexes[i];
            i -= this.starts[volIndex];

            this.volumes[volIndex].SetWeight(i, v);
        }

        public void SetWeightGradient(int i, double v)
        {
            var volIndex = this.indexes[i];
            i -= this.starts[volIndex];

            this.volumes[volIndex].SetWeightGradient(i, v);
        }

        public void ZeroGradients()
        {
            foreach(var vol in this.volumes)
            {
                vol.ZeroGradients();
            }
        }

        public IEnumerator<double> GetEnumerator()
        {
            for (int i = 0; i < this.Length; i++)
            {
                yield return this.GetWeight(i);
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
    }
}