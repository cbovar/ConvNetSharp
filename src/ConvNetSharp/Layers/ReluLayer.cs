using System;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace ConvNetSharp.Layers
{
    /// <summary>
    ///     Implements ReLU nonlinearity elementwise
    ///     x -> max(0, x)
    ///     the output is in [0, inf)
    /// </summary>
    [DataContract]
    [Serializable]
    public class ReluLayer : LayerBase
    {
        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var output = input.Clone();

#if PARALLEL
            Parallel.For(0, input.Length, i =>
#else
            for (var i = 0; i < length; i++)
#endif
            {
                if (output.GetWeight(i) < 0)
                {
                    output.SetWeight(i, 0); // threshold at 0
                }
            }
#if PARALLEL
                );
#endif
            this.OutputActivation = output;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation; // we need to set dw of this
            var outputActivation = this.OutputActivation;
            var length = volume.Length;
            volume.ZeroGradients(); // zero out gradient wrt data

#if PARALLEL
            Parallel.For(0, length, i =>
#else
            for (var i = 0; i < length; i++)
#endif
            {
                if (outputActivation.GetWeight(i) <= 0)
                {
                    volume.SetWeightGradient(i, 0); // threshold
                }
                else
                {
                    volume.SetWeightGradient(i, outputActivation.GetWeightGradient(i));
                }
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