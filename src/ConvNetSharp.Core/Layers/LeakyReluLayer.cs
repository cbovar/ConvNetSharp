using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    /// <summary>
    ///     Implements LeakyReLU nonlinearity elementwise
    ///     x -> x > 0, x, otherwise 0.01x
    ///     the output is in [0, inf)
    /// </summary>
    public class LeakyReluLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public LeakyReluLayer()
        {

        }

        public LeakyReluLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;

            this.OutputActivation.DoLeakyReluGradient(this.InputActivation,
                this.OutputActivationGradients,
                this.InputActivationGradients);
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.DoLeakyRelu(this.OutputActivation);
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