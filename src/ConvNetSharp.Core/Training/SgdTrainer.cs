using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Training
{
    /// <summary>
    ///     Stochastic gradient descent
    /// TODO: L1 regularization , L1DecayLoss, L2DecayLoss
    /// </summary>
    public class SgdTrainer<T> : TrainerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        // last iteration gradients (used for momentum calculations)
        private readonly List<Volume<T>> gsum = new List<Volume<T>>();

        public SgdTrainer(INet<T> net) : base(net)
        {
        }

        public T L1Decay { get; set; }

        public T L2Decay { get; set; }

        public T Momentum { get; set; }

        public T L2DecayLoss { get; private set; }

        public T L1DecayLoss { get; private set; }

        public T LearningRate { get; set; }

        protected override void Backward(Volume<T> y)
        {
            base.Backward(y);

            this.L2DecayLoss = Ops<T>.Zero;
            this.L1DecayLoss = Ops<T>.Zero;
        }

        protected override void TrainImplem()
        {
            var parametersAndGradients = this.Net.GetParametersAndGradients();
            var isMomentumGreaterThanZero = Ops<T>.GreaterThan(this.Momentum, Ops<T>.Zero);

            // initialize lists for accumulators. Will only be done once on first iteration
            if (this.gsum.Count == 0 && isMomentumGreaterThanZero)
            {
                foreach (var t in parametersAndGradients)
                {
                    this.gsum.Add(BuilderInstance<T>.Volume.SameAs(t.Volume.Shape));
                }
            }

            T factor = Ops<T>.Divide(this.LearningRate, Ops<T>.Cast(this.BatchSize));

            // perform an update for all sets of weights
            for (var i = 0; i < parametersAndGradients.Count; i++)
            {
                var parametersAndGradient = parametersAndGradients[i];
                var vol = parametersAndGradient.Volume;
                var grad = parametersAndGradient.Gradient;

                // learning rate for some parameters.
                var l2DecayMul = parametersAndGradient.L2DecayMul ?? Ops<T>.One;
                var l1DecayMul = parametersAndGradient.L1DecayMul ?? Ops<T>.One;
                var l2Decay = Ops<T>.Multiply(this.L2Decay, l2DecayMul);
                var l1Decay = Ops<T>.Multiply(this.L1Decay, l1DecayMul);

                //  this.L2DecayLoss += l2Decay * vol.Get(j) * vol.Get(j) / 2; // accumulate weight decay loss
                //  this.L1DecayLoss += l1Decay * Math.Abs(vol.Get(j));
                //  var l1Grad = l1Decay * (vol.Get(j) > 0 ? 1 : -1);

                var l2Grad = vol*l2Decay;
                var gij = grad + l2Grad;

                //  var gij = (l2Grad + l1Grad + vol.GetGradient(j)) / this.BatchSize; // raw batch gradient

                if (isMomentumGreaterThanZero)
                {
                    // momentum update
                    var dx = this.gsum[i]*this.Momentum + gij*factor; // step
                    this.gsum[i] = (Volume<T>) dx.Clone(); // back this up for next iteration of momentum
                    vol.MapInplace((v, d) => d, vol - dx); // apply corrected gradient
                }
                else
                {
                    // vanilla sgd
                    vol.MapInplace((v, d) => d, vol - gij*factor);
                }

                grad.Clear(); // zero out gradient so that we can begin accumulating anew
            }
        }
    }
}