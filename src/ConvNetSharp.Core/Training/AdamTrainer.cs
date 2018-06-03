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

        public T LearningRate { get; set; }

        public T Eps { get; set; }

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

            // perform an update for all sets of weights
            for (var i = 0; i < parametersAndGradients.Count; i++)
            {
                var parametersAndGradient = parametersAndGradients[i];
                var vol = parametersAndGradient.Volume;
                var grad = parametersAndGradient.Gradient;

                grad.Multiply(Ops<T>.Divide(Ops<T>.One, Ops<T>.Cast(this.BatchSize)), grad); // grad *= 1 / BatchSize

                using (var temp1 = BuilderInstance<T>.Volume.SameAs(vol.Shape))
                using (var temp2 = BuilderInstance<T>.Volume.SameAs(vol.Shape))
                using (var gradgrad = BuilderInstance<T>.Volume.SameAs(vol.Shape))
                using (var two = BuilderInstance<T>.Volume.From(new[] { Ops<T>.Cast(2.0) }, new Shape(1)))
                using (var epsilon = BuilderInstance<T>.Volume.From(new[] { this.Eps }, new Shape(1)))
                {
                    // momentum update

                    // update biased first moment estimate: gsum[i] = gsum[i] * Beta1 +  (1 - Beta1) * grad
                    this.gsum[i].Multiply(this.Beta1, temp1); // temp1 = this.gsum[i] * this.Beta1
                    grad.Multiply(Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(this.Beta1)), this.gsum[i]); //  this.gsum[i] =  grad * (1 - Beta1)
                    temp1.Add(this.gsum[i]); //  this.gsum[i] += temp1

                    grad.Power(two, gradgrad); // gradgrad = grad * grad

                    // update biased second moment estimate: xsum[i] = xsum[i] * Beta2 +  (1 - Beta2) * grad * grad
                    this.xsum[i].Multiply(this.Beta2, temp1); // temp1 = this.xsum[i] * this.Beta2
                    gradgrad.Multiply(Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(this.Beta2)), this.xsum[i]); // temp2 =  gradgrad * (1 - Beta2)
                    temp1.Add(this.xsum[i]); //  this.xsum[i] += temp1

                    var biasCorr1 = temp1;
                    var biasCorr2 = temp2;

                    this.gsum[i].Multiply(Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(Ops<T>.Pow(this.Beta1, Ops<T>.Cast(this.k)))), biasCorr1); // correct bias first moment estimate
                    this.xsum[i].Multiply(Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(Ops<T>.Pow(this.Beta2, Ops<T>.Cast(this.k)))), biasCorr2); // correct bias second moment estimate

                    biasCorr2.Sqrt(biasCorr2); // biasCorr2 = sqrt(biasCorr2)
                    epsilon.Add(biasCorr2); // biasCorr2 += epsilon

                    var dx = biasCorr1;
                    dx.Multiply(this.LearningRate, dx);
                    dx.Divide(biasCorr2, dx);

                    dx.SubtractFrom(vol, vol);
                }

                grad.Clear(); // zero out gradient so that we can begin accumulating anew


                this.k += this.BatchSize;
            }
        }
    }
}