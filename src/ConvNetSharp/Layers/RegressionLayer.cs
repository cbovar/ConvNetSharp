using System;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    /// <summary>
    ///     implements an L2 regression cost layer,
    ///     so penalizes \sum_i(||x_i - y_i||^2), where x is its input
    ///     and y is the user-provided array of "correct" values.
    /// </summary>
    [DataContract]
    [Serializable]
    public class RegressionLayer : LastLayerBase, ILastLayer
    {
        public RegressionLayer()
        {
        }

        public override double Backward(double y)
        {
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.InputActivation;
            x.ZeroGradients(); // zero out the gradient of input Vol
            var loss = 0.0;

            // lets hope that only one number is being regressed
            var dy = x.Get(0) - y;
            x.SetGradient(0, dy);
            loss += 0.5 * dy * dy;

            return loss;
        }

        public override double Backward(double[] y)
        {
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.InputActivation;
            x.ZeroGradients(); // zero out the gradient of input Vol
            var loss = 0.0;

            for (var i = 0; i < this.OutputDepth; i++)
            {
                var dy = x.Get(i) - y[i];
                x.SetGradient(i,  dy);
                loss += 0.5 * dy * dy;
            }

            return loss;
        }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;
            this.OutputActivation = input;
            return input; // identity function
        }

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            var inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputDepth = inputCount;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
        }
    }
}