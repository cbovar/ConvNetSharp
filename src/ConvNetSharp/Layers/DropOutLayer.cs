using System;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    [DataContract]
    [Serializable]
    public class DropOutLayer : LayerBase
    {
        private static readonly Random Random = new Random(RandomUtilities.Seed);

        [DataMember]
        private bool[] dropped;

        public DropOutLayer(double dropProb = 0.5)
        {
            this.DropProb = dropProb;
        }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var output = input.Clone();
            var length = input.Weights.Length;

            if (isTraining)
            {
                // do dropout
                for (var i = 0; i < length; i++)
                {
                    if (Random.NextDouble() < this.DropProb.Value)
                    {
                        output.Weights[i] = 0;
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
                    output.Weights[i] *= 1 - this.DropProb.Value;
                }
            }

            this.OutputActivation = output;
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
                if (!this.dropped[i])
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

            this.dropped = new bool[this.OutputWidth * this.OutputHeight * this.OutputDepth];
        }
    }
}