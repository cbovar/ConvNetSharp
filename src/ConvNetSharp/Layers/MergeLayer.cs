using System;
using System.Linq;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    [DataContract]
    [Serializable]
    public class MergeLayer : LayerBase
    {
        public MergeLayer(params LayerBase[] layers)
        {
            var total = 0;
            foreach (var layer in layers)
            {
                this.Parents.Add(layer);
                total += layer.OutputDepth * layer.OutputHeight * layer.OutputWidth;
            }

            this.Init(1, 1, total);
        }

        public override void Backward()
        {
            var volume = this.InputActivation; // we need to set dw of this
            var chainGradient = this.OutputActivation;
            var length = volume.Length;
            volume.ZeroGradients(); // zero out gradient wrt data

            for (var i = 0; i < length; i++)
            {
                volume.SetGradient(i, chainGradient.GetGradient(i)); // copy over the gradient
            }
        }

        public override IVolume Forward(bool isTraining)
        {
            // Simple implementation for the moment:
            var vol = new VolumeWrapper(this.Parents.Select(o => o.Forward(isTraining)));
            this.InputActivation = vol;
            this.OutputActivation = vol.Clone();

            return this.OutputActivation;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.OutputWidth = 1;
            this.OutputHeight = 1;
            this.OutputDepth = inputWidth * InputHeight * InputDepth;
        }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            throw new NotImplementedException();
        }
    }
}
