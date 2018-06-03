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

        /// <summary>
        /// This computes the cross entropy loss and its gradient (not the softmax gradient)
        /// </summary>
        /// <param name="y"></param>
        /// <param name="loss"></param>
        public override void Backward(Volume<T> y, out T loss)
        {
            // input gradient = pi - yi
            y.SubtractFrom(this.OutputActivation, this.InputActivationGradients.ReShape(this.OutputActivation.Shape.Dimensions));

            //loss is the class negative log likelihood
            loss = Ops<T>.Zero;
            for (var n = 0; n < y.Shape.Dimensions[3]; n++)
            {
                for (var d = 0; d < y.Shape.Dimensions[2]; d++)
                {
                    for (var h = 0; h < y.Shape.Dimensions[1]; h++)
                    {
                        for (var w = 0; w < y.Shape.Dimensions[0]; w++)
                        {
                            var expected = y.Get(w, h, d, n);
                            var actual = this.OutputActivation.Get(w, h, d, n);
                            if (Ops<T>.Zero.Equals(actual))
                                actual = Ops<T>.Epsilon;
                            var current = Ops<T>.Multiply(expected, Ops<T>.Log(actual));

                            loss = Ops<T>.Add(loss, current);
                        }
                    }
                }
            }

            loss = Ops<T>.Negate(loss);

            if (Ops<T>.IsInvalid(loss))
                throw new ArgumentException("Error during calculation!");
        }

        public override void Backward(Volume<T> outputGradient)
        {
            throw new NotImplementedException();
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.Softmax(this.OutputActivation);
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