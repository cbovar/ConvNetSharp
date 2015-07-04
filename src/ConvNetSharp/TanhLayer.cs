using System;
using System.Runtime.Serialization;

namespace ConvNetSharp
{
    [DataContract]
    public class TanhLayer : LayerBase
    {
        public override Volume Forward(Volume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var outputActivation = input.CloneAndZero();
            var length = input.Weights.Length;

            for (var i = 0; i < length; i++)
            {
                outputActivation.Weights[i] = Math.Tanh(input.Weights[i]);
            }

            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation; // we need to set dw of this
            var volume2 = this.OutputActivation;
            var length = volume.Weights.Length;
            volume.WeightGradients = new double[length]; // zero out gradient wrt data

            for (var i = 0; i < length; i++)
            {
                var v2wi = volume2.Weights[i];
                volume.WeightGradients[i] = (1.0 - v2wi * v2wi) * volume2.WeightGradients[i];
            }
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.OutputDepth = inputDepth;
            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;
        }
    }
}