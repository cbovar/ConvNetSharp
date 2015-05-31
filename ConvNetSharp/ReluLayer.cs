namespace ConvNetSharp
{
    /// <summary>
    ///     Implements ReLU nonlinearity elementwise
    ///     x -> max(0, x)
    ///     the output is in [0, inf)
    /// </summary>
    public class ReluLayer : LayerBase
    {
        public override Volume Forward(Volume volume, bool isTraining = false)
        {
            this.InputActivation = volume;
            var volume2 = volume.Clone();
            var length = volume.Weights.Length;
            double[] v2W = volume2.Weights;

            for (var i = 0; i < length; i++)
            {
                if (v2W[i] < 0)
                {
                    v2W[i] = 0; // threshold at 0
                }
            }

            this.OutputActivation = volume2;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation; // we need to set dw of this
            var v2 = this.OutputActivation;
            var length = volume.Weights.Length;
            volume.WeightGradients = new double[length]; // zero out gradient wrt data

            for (var i = 0; i < length; i++)
            {
                if (v2.Weights[i] <= 0)
                {
                    volume.WeightGradients[i] = 0; // threshold
                }
                else
                {
                    volume.WeightGradients[i] = v2.WeightGradients[i];
                }
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