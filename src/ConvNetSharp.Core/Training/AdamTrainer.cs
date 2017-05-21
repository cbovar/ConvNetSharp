using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Training
{
    public class AdamTrainer<T> : TrainerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly List<Volume<T>> gsum = new List<Volume<T>>(); // last iteration gradients (used for momentum calculations)
        private readonly List<Volume<T>> xsum = new List<Volume<T>>();
        private int k;

        public AdamTrainer(INet<T> net) : base(net)
        {
            if (typeof(T) == typeof(double))
            {
                this.Eps = (T)(ValueType)1e-8;
            }
            else if (typeof(T) == typeof(float))
            {
                this.Eps = (T)(ValueType)(float)1e-8;
            }
        }

        public T Beta1 { get; set; }

        public T Beta2 { get; set; }

        public T L1Decay { get; set; }

        public T L2Decay { get; set; }

        public T L2DecayLoss { get; private set; }

        public T L1DecayLoss { get; private set; }

        public T LearningRate { get; set; }

        public T Eps { get; set; }

        protected override void Backward(Volume<T> y)
        {
            base.Backward(y);

            this.L2DecayLoss = Ops<T>.Zero;
            this.L1DecayLoss = Ops<T>.Zero;
        }

        protected override void TrainImplem()
        {
            var parametersAndGradients = this.Net.GetParametersAndGradients();

            // initialize lists for accumulators. Will only be done once on first iteration
            if (this.gsum.Count == 0)
            {
                foreach (var t in parametersAndGradients)
                {
                    this.gsum.Add(BuilderInstance<T>.Volume.SameAs(t.Volume.Shape));
                    this.xsum.Add(BuilderInstance<T>.Volume.SameAs(t.Volume.Shape));
                }
            }

            var factor = Ops<T>.Divide(Ops<T>.One, Ops<T>.Cast(this.BatchSize));

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

                var l1Grad = vol.Clone();
                l1Grad.MapInplace(x => Ops<T>.GreaterThan(x, Ops<T>.Zero) ? Ops<T>.One : Ops<T>.Negate(Ops<T>.One));
                l1Grad = l1Grad * l1Decay;

                var l2Grad = vol * l2Decay;

                var gij = (grad + l2Grad + l1Grad) * factor;

                // momentum update
                this.gsum[i] = this.gsum[i] * this.Beta1 + gij * Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(this.Beta1)); // update biased first moment estimate
                var gijgij = gij.Clone();
                gijgij.MapInplace(x => Ops<T>.Multiply(x, x));
                this.xsum[i] = this.xsum[i] * this.Beta2 + gijgij * Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(this.Beta2)); // update biased second moment estimate
                var biasCorr1 = this.gsum[i] * Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(Ops<T>.Pow(this.Beta1, Ops<T>.Cast(this.k)))); // correct bias first moment estimate
                var biasCorr2 = this.xsum[i] * Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(Ops<T>.Pow(this.Beta2, Ops<T>.Cast(this.k)))); // correct bias second moment estimate
                biasCorr2.MapInplace(x => Ops<T>.Add(Ops<T>.Sqrt(x), this.Eps));

                var dx = biasCorr1 * this.LearningRate;
                dx.MapInplace((l, r) => Ops<T>.Divide(l, r), biasCorr2);

                vol.MapInplace((v, d) => d, vol - dx); // apply corrected gradient

                grad.Clear(); // zero out gradient so that we can begin accumulating anew
            }

            this.k += this.BatchSize;
        }
    }
}