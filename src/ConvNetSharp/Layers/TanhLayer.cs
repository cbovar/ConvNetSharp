using System;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace ConvNetSharp.Layers
{
    [DataContract]
    [Serializable]
    public class TanhLayer : LayerBase
    {
        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var outputActivation = input.CloneAndZero();
            var length = input.Length;

#if PARALLEL
            Parallel.For(0, length, i =>
#else
            for (var i = 0; i < length; i++)
#endif
            { outputActivation.Set(i, Math.Tanh(input.Get(i))); }
#if PARALLEL
                );
#endif
            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation; // we need to set dw of this
            var volume2 = this.OutputActivation;
            var length = volume.Length;
            volume.ZeroGradients(); // zero out gradient wrt data

#if PARALLEL
            Parallel.For(0, length, i =>
#else
            for (var i = 0; i < length; i++)
#endif
            {
                var v2wi = volume2.Get(i);
                volume.SetGradient(i, (1.0 - v2wi * v2wi) * volume2.GetGradient(i));
            }
#if PARALLEL
                );
#endif
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