using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class DropoutLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public DropoutLayer(Dictionary<string, object> data) : base(data)
        {
            this.DropProbability = (T)Convert.ChangeType(data["DropProbability"], typeof(T));
        }

        public DropoutLayer(T dropProbability)
        {
            this.DropProbability = dropProbability;
        }

        public T DropProbability { get; set; }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;

            this.InputActivationGradients.Clear();

            this.OutputActivation.DropoutGradient(this.InputActivation, this.OutputActivationGradients, this.InputActivationGradients, this.DropProbability);
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.Dropout(isTraining ? this.DropProbability : Ops<T>.Zero, this.OutputActivation);
            return this.OutputActivation;
        }

        public override Dictionary<string, object> GetData()
        {
            var dico = base.GetData();
            dico["DropProbability"] = this.DropProbability;
            return dico;
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