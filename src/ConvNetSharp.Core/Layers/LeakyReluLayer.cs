using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    /// <summary>
    ///     Implements LeakyReLU nonlinearity elementwise
    ///     x -> x > 0, x, otherwise alpha * x
    /// </summary>
    public class LeakyReluLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public LeakyReluLayer(T alpha)
        {
            this.Alpha = alpha;
        }

        public LeakyReluLayer(Dictionary<string, object> data) : base(data)
        {
            this.Alpha = (T)Convert.ChangeType(data["Alpha"], typeof(T));
        }

        public T Alpha { get; set; }

        public override Dictionary<string, object> GetData()
        {
            var dico = base.GetData();

            dico["Alpha"] = this.Alpha;

            return dico;
        }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;
            this.OutputActivation.LeakyReluGradient(this.OutputActivationGradients, this.InputActivationGradients, this.Alpha);
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.LeakyRelu(this.Alpha, this.OutputActivation);
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