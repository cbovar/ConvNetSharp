using System;

namespace ConvNetSharp
{
    public class TanhLayer : LayerBase
    {
        public override Volume Forward(Volume volume, bool isTraining = false)
        {
            this.InputActivation = volume;
            var outputActivation = volume.CloneAndZero();
            var length = volume.Weights.Length;

            for (var i = 0; i < length; i++)
            {
                outputActivation.Weights[i] = Math.Tanh(volume.Weights[i]);
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
                volume.WeightGradients[i] = (1.0 - v2wi*v2wi)*volume2.WeightGradients[i];
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