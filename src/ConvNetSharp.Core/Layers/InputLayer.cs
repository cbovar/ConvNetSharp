using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class InputLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public InputLayer(Dictionary<string, object> data) : base(data)
        {
            this.OutputWidth = this.InputWidth;
            this.OutputHeight = this.InputHeight;
            this.OutputDepth = this.InputDepth;
        }

        public InputLayer(int inputWidth, int inputHeight, int inputDepth)
        {
            Init(inputWidth, inputHeight, inputDepth);

            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;
            this.OutputDepth = inputDepth;
        }

        public override void Backward(Volume<T> outputGradient)
        {
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            this.OutputActivation = input;
            return this.OutputActivation;
        }

        public override Volume<T> Forward(bool isTraining)
        {
            return this.OutputActivation;
        }
    }
}