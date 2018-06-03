using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    /// <summary>
    ///     implements an L2 regression cost layer,
    ///     so penalizes \sum_i(||x_i - y_i||^2), where x is its input
    ///     and y is the user-provided array of "correct" values.
    ///     Input should have a shape of [1, 1, 1, n] where n is the batch size
    /// </summary>
    public class RegressionLayer<T> : LastLayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> _result;
        private Volume<T> _sum;

        public RegressionLayer()
        {
        }

        public RegressionLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;
            this.OutputDepth = inputDepth;
        }

        public override void Backward(Volume<T> outputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Backward(Volume<T> y, out T loss)
        {
            var reshape = y.ReShape(new Shape(1, 1, -1, Shape.Keep));
            var dy = this.InputActivationGradients.ReShape(this.OutputActivation.Shape.Dimensions);
            reshape.SubtractFrom(this.OutputActivation, dy);

            if (this._result == null)
            {
                this._result = BuilderInstance<T>.Volume.SameAs(this.OutputActivation.Shape);
                this._sum = BuilderInstance<T>.Volume.SameAs(new Shape(1));
            }

            this._sum.Clear();
            dy.Multiply(dy, this._result); // dy * dy
            var half = (T)Convert.ChangeType(0.5, typeof(T));
            this._result.Multiply(half, this._result); // dy * dy * 0.5
            this._result.Sum(this._sum); // sum over all batch
            var batchSize = y.Shape.Dimensions[3];
            loss = Ops<T>.Divide(this._sum.Get(0), Ops<T>.Cast(batchSize)); // average
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            return input;
        }
    }
}