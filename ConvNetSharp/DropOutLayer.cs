using System;

namespace ConvNetSharp
{
    public class DropOutLayer : LayerBase
    {
        private readonly Random random = new Random();
        private bool[] dropped;

        public DropOutLayer(double dropProb)
        {
            this.DropProb = dropProb;
        }

        public override Volume Forward(Volume volume, bool isTraining = false)
        {
            this.InputActivation = volume;
            var V2 = volume.Clone();
            var length = volume.Weights.Length;

            if (isTraining)
            {
                // do dropout
                for (var i = 0; i < length; i++)
                {
                    if (this.random.NextDouble() < this.DropProb.Value)
                    {
                        V2.Weights[i] = 0;
                        this.dropped[i] = true;
                    } // drop!
                    else
                    {
                        this.dropped[i] = false;
                    }
                }
            }
            else
            {
                // scale the activations during prediction
                for (var i = 0; i < length; i++)
                {
                    V2.Weights[i] *= this.DropProb.Value;
                }
            }

            this.OutputActivation = V2;
            return this.OutputActivation; // dummy identity function for now
        }

        public override void Backward()
        {
            var volume = this.InputActivation; // we need to set dw of this
            var chainGradient = this.OutputActivation;
            var length = volume.Weights.Length;
            volume.WeightGradients = new double[length]; // zero out gradient wrt data

            for (var i = 0; i < length; i++)
            {
                if (!(this.dropped[i]))
                {
                    volume.WeightGradients[i] = chainGradient.WeightGradients[i]; // copy over the gradient
                }
            }
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // computed
            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;
            this.OutputDepth = inputDepth;

            this.DropProb = this.DropProb;
            this.dropped = new bool[this.OutputWidth * this.OutputHeight * this.OutputDepth];
        }
    }
}