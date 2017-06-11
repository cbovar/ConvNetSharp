using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Training
{
    /// <summary>
    ///     Stochastic gradient descent
    /// TODO: L1DecayLoss, L2DecayLoss
    /// </summary>
    public class SgdTrainer<T> : TrainerBase<T>, IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        // last iteration gradients (used for momentum calculations)
        private readonly List<Volume<T>> velocities = new List<Volume<T>>();
        private readonly List<Volume<T>> deltas = new List<Volume<T>>();
        private readonly List<Volume<T>> regGrads = new List<Volume<T>>();

        public SgdTrainer(INet<T> net) : base(net)
        {
        }

        public T L1Decay { get; set; }

        public T L2Decay { get; set; }

        public T Momentum { get; set; }

        public T L2DecayLoss { get; private set; }

        public T L1DecayLoss { get; private set; }

        public T LearningRate { get; set; }

        public void Dispose()
        {
            foreach (var v in velocities)
                v.Dispose();
            foreach (var d in deltas)
                d.Dispose();
            foreach (var r in regGrads)
                r.Dispose();
        }

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
            if (this.velocities.Count == 0)
            {
                foreach (var parameter in parametersAndGradients)
                {
                    this.velocities.Add(BuilderInstance<T>.Volume.SameAs(parameter.Volume.Shape));
                    this.deltas.Add(BuilderInstance<T>.Volume.SameAs(parameter.Volume.Shape));
                    this.regGrads.Add(BuilderInstance<T>.Volume.SameAs(parameter.Volume.Shape));
                }
            }

            // perform an update for all sets of weights
            for (var i = 0; i < parametersAndGradients.Count; i++)
            {
                var parametersAndGradient = parametersAndGradients[i];
                var parameters = parametersAndGradient.Volume;
                var gradients = parametersAndGradient.Gradient;
                var delta = this.deltas[i];
                var regularizationGradients = this.regGrads[i];
                var velocity = this.velocities[i];

                // learning rate for some parameters.
                var l2DecayMul = parametersAndGradient.L2DecayMul ?? Ops<T>.One;
                var l1DecayMul = parametersAndGradient.L1DecayMul ?? Ops<T>.One;
                var l2Decay = Ops<T>.Multiply(this.L2Decay, l2DecayMul);
                var l1Decay = Ops<T>.Multiply(this.L1Decay, l1DecayMul);

                //  this.L2DecayLoss += l2Decay * vol.Get(j) * vol.Get(j) / 2; // accumulate weight decay loss
                //  this.L1DecayLoss += l1Decay * Math.Abs(vol.Get(j));

                //L1 regularization
                if (Ops<T>.GreaterThan(l1Decay, Ops<T>.Zero))
                {
                    //l1Grad = l1Grad * l1Decay;
                    parameters.Storage.Map(x => Ops<T>.GreaterThan(x, Ops<T>.Zero) ? Ops<T>.One : Ops<T>.Negate(Ops<T>.One), regularizationGradients.Storage);
                    regularizationGradients.DoMultiply(delta, l1Decay);
                }
                else
                {
                    delta.Clear();
                }

                //L2 regularization
                if (Ops<T>.GreaterThan(l2Decay, Ops<T>.Zero))
                {
                    //l2Grad = vol * l2Decay;
                    parameters.DoMultiply(regularizationGradients, l2Decay);
                    delta.DoAdd(regularizationGradients, delta);
                }

                T batchAdjustedLearningRate = Ops<T>.Divide(this.LearningRate, Ops<T>.Cast(this.BatchSize));

                //delta = gradient + regularization;
                gradients.DoMultiply(gradients, batchAdjustedLearningRate);
                delta.DoMultiply(delta, this.LearningRate);
                delta.DoAdd(gradients, delta);

                if (isMomentumGreaterThanZero)
                {
                    // sgd with momentum update
                    velocity.DoMultiply(velocity, this.Momentum);    // step
                    velocity.DoAdd(delta, velocity);
                    velocity.DoSubtractFrom(parameters, parameters); // apply corrected gradient
                }
                else
                {
                    // vanilla sgd
                    delta.DoSubtractFrom(parameters, parameters);
                }

                // zero out gradient so that we can begin accumulating anew
                gradients.Clear();
            }
        }
    }
}