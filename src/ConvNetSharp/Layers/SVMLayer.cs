using System;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    [DataContract]
    [Serializable]
    public class SvmLayer : LastLayerBase, ILastLayer, IClassificationLayer
    {
        public SvmLayer(int classCount)
        {
            this.ClassCount = classCount;
        }

        [DataMember]
        public int ClassCount { get; set; }

        public override double Backward(double yd)
        {
            var y = (int)yd;
            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.InputActivation;
            x.ZeroGradients(); // zero out the gradient of input Vol

            // we're using structured loss here, which means that the score
            // of the ground truth should be higher than the score of any other 
            // class, by a margin
            var yscore = x.Get(y); // score of ground truth
            const double margin = 1.0;
            var loss = 0.0;
            for (var i = 0; i < this.OutputDepth; i++)
            {
                if (y == i)
                {
                    continue;
                }
                var ydiff = -yscore + x.Get(i) + margin;
                if (ydiff > 0)
                {
                    // violating dimension, apply loss
                    x.SetGradient(i, x.GetGradient(i) + 1);
                    x.SetGradient(y, x.GetGradient(y) - 1);
                    loss += ydiff;
                }
            }

            return loss;
        }

        public override double Backward(double[] y)
        {
            throw new NotImplementedException();
        }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;
            this.OutputActivation = input; // nothing to do, output raw scores
            return input;
        }

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // computed
            this.OutputDepth = inputWidth * inputHeight * inputDepth;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
        }
    }
}