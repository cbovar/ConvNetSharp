﻿using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class DropoutLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public T DropProbability { get; set; }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;

            this.InputActivationGradients.Clear();

            this.OutputActivation.DoDropoutGradient(this.InputActivation, this.OutputActivationGradients, this.InputActivationGradients);
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.DoDropout(this.OutputActivation, isTraining, this.DropProbability);
            return this.OutputActivation;
        }
    }
}