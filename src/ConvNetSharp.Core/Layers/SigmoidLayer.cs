using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class SigmoidLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public SigmoidLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public SigmoidLayer()
        {
        }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;
            this.OutputActivation.DoSigmoidGradient(this.InputActivation, this.OutputActivationGradients, this.InputActivationGradients);
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.DoSigmoid(this.OutputActivation);
            return this.OutputActivation;
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