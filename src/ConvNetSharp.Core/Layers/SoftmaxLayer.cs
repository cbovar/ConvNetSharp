using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class SoftmaxLayer<T> : LastLayerBase<T>, IClassificationLayer where T : struct, IEquatable<T>, IFormattable
    {
        public SoftmaxLayer(Dictionary<string, object> data) : base(data)
        {
            this.ClassCount = Convert.ToInt32(data["ClassCount"]);
        }

        public SoftmaxLayer(int classCount)
        {
            this.ClassCount = classCount;
        }

        public int ClassCount { get; set; }

        public override void Backward(Volume<T> y, out T loss)
        {
            this.OutputActivation.DoSoftMaxGradient(this.OutputActivation - y, this.InputActivationGradients);

            // loss is the class negative log likelihood
            loss = Ops<T>.Zero;
            for (var n = 0; n < y.Shape.GetDimension(3); n++)
            {
                for (var i = 0; i < y.Shape.GetDimension(2); i++)
                {
                    if (!y.Get(0, 0, i, n).Equals(Ops<T>.Zero))
                    {
                        loss = Ops<T>.Add(loss, Ops<T>.Log(this.OutputActivation.Get(0, 0, i, n)));
                    }
                }
            }

            loss = Ops<T>.Negate(loss);
        }

        public override void Backward(Volume<T> outputGradient)
        {
            throw new NotImplementedException();
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.DoSoftMax(this.OutputActivation);
            return this.OutputActivation;
        }

        public override Dictionary<string, object> GetData()
        {
            var dico = base.GetData();
            dico["ClassCount"] = this.ClassCount;
            return dico;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            var inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
            this.OutputDepth = inputCount;
        }
    }
}