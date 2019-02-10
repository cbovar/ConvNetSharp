using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Training
{
    /// <summary>
    ///     Stochastic gradient descent
    /// </summary>
    public class SgdTrainer<T> : TrainerBase<T>, IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private readonly List<Volume<T>> regGrads = new List<Volume<T>>();

        // last iteration gradients (used for momentum calculations)
        private readonly List<Volume<T>> velocities = new List<Volume<T>>();

        public SgdTrainer(INet<T> net) : base(net)
        {
        }

        public T Momentum { get; set; }

        public T LearningRate { get; set; }

        public void Dispose()
        {
            foreach (var v in this.velocities)
            {
                v.Dispose();
            }

            foreach (var r in this.regGrads)
            {
                r.Dispose();
            }
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
                    this.regGrads.Add(BuilderInstance<T>.Volume.SameAs(parameter.Volume.Shape));
                }
            }

            // perform an update for all sets of weights
            for (var i = 0; i < parametersAndGradients.Count; i++)
            {
                var parametersAndGradient = parametersAndGradients[i];
                var parameters = parametersAndGradient.Volume;
                var gradients = parametersAndGradient.Gradient;
                var velocity = this.velocities[i];

                var batchAdjustedLearningRate = Ops<T>.Divide(this.LearningRate, Ops<T>.Cast(this.BatchSize));

                // delta = gradient + regularization;
                gradients.Multiply(batchAdjustedLearningRate, gradients);

                if (isMomentumGreaterThanZero)
                {
                    // sgd with momentum update
                    velocity.Multiply(this.Momentum, velocity); // step
                    velocity.Add(gradients, velocity);
                    velocity.SubtractFrom(parameters, parameters); // apply corrected gradient
                }
                else
                {
                    // vanilla sgd
                    gradients.SubtractFrom(parameters, parameters);
                }

                // zero out gradient so that we can begin accumulating anew
                gradients.Clear();
            }
        }
    }
}