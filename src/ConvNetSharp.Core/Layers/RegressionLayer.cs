﻿using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    /// <summary>
    ///     implements an L2 regression cost layer,
    ///     so penalizes \sum_i(||x_i - y_i||^2), where x is its input
    ///     and y is the user-provided array of "correct" values.
    /// 
    ///     Input should have a shape of [1, 1, 1, n] where n is the batch size
    /// </summary>
    public class RegressionLayer<T> : LastLayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        Volume<T> _result;
        Volume<T> _sum;

        public override void Backward(Volume<T> outputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Backward(Volume<T> y, out T loss)
        {
            y.DoSubtractFrom(this.OutputActivation, this.InputActivationGradients.ReShape(this.OutputActivation.Shape.Dimensions.ToArray()));

            if (this._result == null)
            {
                this._result = BuilderInstance<T>.Volume.SameAs(this.OutputActivation.Shape);
                this._sum = BuilderInstance<T>.Volume.SameAs(new Shape(1));
            }

            this._sum.Clear();
            this.OutputActivation.DoMultiply(this.OutputActivation, this._result); // dy * dy
            var half = (T)Convert.ChangeType(0.5, typeof(T));
            this._result.DoMultiply(this._result, half); // dy * dy * 0.5
            this._result.DoSum(this._sum); // sum over all batch
            var batchSize = y.Shape.GetDimension(3);
            loss = Ops<T>.Divide(this._sum.Get(0), Ops<T>.Cast(batchSize)); // average
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            return input;
        }
    }
}