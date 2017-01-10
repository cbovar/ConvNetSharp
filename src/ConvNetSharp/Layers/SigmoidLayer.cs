using System;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    /// <summary>
    ///     Implements Sigmoid nnonlinearity elementwise
    ///     x -> 1/(1+e^(-x))
    ///     so the output is between 0 and 1.
    /// </summary>
    [DataContract]
    [Serializable]
    public class SigmoidLayer : LayerBase
    {
        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var volume2 = input.CloneAndZero();
            var length = input.Length;

            for (var i = 0; i < length; i++)
            {
                volume2.Set(i, 1.0 / (1.0 + Math.Exp(-input.Get(i))));
            }

            this.OutputActivation = volume2;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation; // we need to set dw of this
            var volume2 = this.OutputActivation;
            volume.ZeroGradients(); // zero out gradient wrt data

            for (var i = 0; i < volume.Length; i++)
            {
                var v2wi = volume2.Get(i);
                volume.SetGradient(i, v2wi * (1.0 - v2wi) * volume2.GetGradient(i));
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