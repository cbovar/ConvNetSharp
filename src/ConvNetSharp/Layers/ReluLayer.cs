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
        public override Volume Forward(Volume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var output = input.Clone();
            var length = input.Weights.Length;
            double[] outputWeights = output.Weights;

#if PARALLEL
            Parallel.For(0, length, i =>
#else
            for (var i = 0; i < length; i++)
#endif
            {
                if (outputWeights[i] < 0)
                {
                    outputWeights[i] = 0; // threshold at 0
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
            var length = volume.Weights.Length;
            volume.WeightGradients = new double[length]; // zero out gradient wrt data

#if PARALLEL
            Parallel.For(0, length, i =>
#else
            for (var i = 0; i < length; i++)
#endif
            {
                if (outputActivation.Weights[i] <= 0)
                {
                    volume.WeightGradients[i] = 0; // threshold
                }
                else
                {
                    volume.WeightGradients[i] = outputActivation.WeightGradients[i];
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