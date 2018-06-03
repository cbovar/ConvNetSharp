using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    /// <summary>
    ///     Implements ReLU nonlinearity elementwise
    ///     x -> max(0, x)
    ///     the output is in [0, inf)
    /// </summary>
    public class ReluLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public ReluLayer()
        {
            
        }

        public ReluLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;

            this.OutputActivation.ReluGradient(this.InputActivation,
                this.OutputActivationGradients,
                this.InputActivationGradients);
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.Relu(this.OutputActivation);
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