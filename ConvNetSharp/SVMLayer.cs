using System;

namespace ConvNetSharp
{
    public class SvmLayer : LayerBase, IClassificationLayer
    {
        private int inputCount;

        public int ClassCount { get; set; }

        public override Volume Forward(Volume volume, bool isTraining = false)
        {
            this.InputActivation = volume;
            this.OutputActivation = volume; // nothing to do, output raw scores
            return volume;
        }

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // computed
            this.inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputDepth = this.inputCount;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
        }

        public double Backward(int y)
        {
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.InputActivation;
            x.WeightGradients = new double[x.Weights.Length]; // zero out the gradient of input Vol

            // we're using structured loss here, which means that the score
            // of the ground truth should be higher than the score of any other 
            // class, by a margin
            var yscore = x.Weights[y]; // score of ground truth
            const double margin = 1.0;
            var loss = 0.0;
            for (var i = 0; i < this.OutputDepth; i++)
            {
                if (y == i)
                {
                    continue;
                }
                var ydiff = -yscore + x.Weights[i] + margin;
                if (ydiff > 0)
                {
                    // violating dimension, apply loss
                    x.WeightGradients[i] += 1;
                    x.WeightGradients[y] -= 1;
                    loss += ydiff;
                }
            }

            return loss;
        }
    }
}