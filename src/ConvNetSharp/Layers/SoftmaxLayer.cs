using System;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    /// <summary>
    ///     This is a classifier, with N discrete classes from 0 to N-1
    ///     it gets a stream of N incoming numbers and computes the softmax
    ///     function (exponentiate and normalize to sum to 1 as probabilities should)
    /// </summary>
    [DataContract]
    [Serializable]
    public class SoftmaxLayer : LayerBase, ILastLayer, IClassificationLayer
    {
        public SoftmaxLayer()
        {
        }

        [DataMember]
        private double[] es;

        public SoftmaxLayer(int classCount)
        {
            this.ClassCount = classCount;
        }

        [DataMember]
        public int ClassCount { get; set; }

        public double Backward(double y)
        {
            var yint = (int)y;

            // compute and accumulate gradient wrt weights and bias of this layer
            var x = this.InputActivation;
            x.WeightGradients = new double[x.Weights.Length]; // zero out the gradient of input Vol

            for (var i = 0; i < this.OutputDepth; i++)
            {
                var indicator = i == yint ? 1.0 : 0.0;
                var mul = -(indicator - this.es[i]);
                x.WeightGradients[i] = mul;
            }

            // loss is the class negative log likelihood
            return -Math.Log(this.es[yint]);
        }

        public double Backward(double[] y)
        {
            throw new NotImplementedException();
        }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            this.InputActivation = input;

            var outputActivation = new Volume(1, 1, this.OutputDepth, 0.0);

            // compute max activation
            double[] temp = input.Weights;
            var amax = input.Weights[0];
            for (var i = 1; i < this.OutputDepth; i++)
            {
                if (temp[i] > amax)
                {
                    amax = temp[i];
                }
            }

            // compute exponentials (carefully to not blow up)
            var es = new double[this.OutputDepth];
            var esum = 0.0;
            for (var i = 0; i < this.OutputDepth; i++)
            {
                var e = Math.Exp(temp[i] - amax);
                esum += e;
                es[i] = e;
            }

            // normalize and output to sum to one
            for (var i = 0; i < this.OutputDepth; i++)
            {
                es[i] /= esum;
                outputActivation.Weights[i] = es[i];
            }

            this.es = es; // save these for backprop
            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            var inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputDepth = inputCount;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
        }

        #region Serialization

        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);

            info.AddValue("ClassCount", this.ClassCount);
            info.AddValue("es", this.es);
        }

        private SoftmaxLayer(SerializationInfo info, StreamingContext context) : base(info, context)
        {
            this.ClassCount = info.GetInt32("ClassCount");
            this.es = (double[])info.GetValue("es", typeof(double[]));
        }

        #endregion
    }
}