using System;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Training
{
    public abstract class TrainerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected readonly INet<T> Net;

        protected TrainerBase(INet<T> net)
        {
            this.Net = net;
        }

        public double BackwardTimeMs { get; private set; }

        public double ForwardTimeMs { get; private set; }

        public double UpdateWeightsTimeMs { get; private set; }

        public virtual T Loss { get; private set; }

        public int BatchSize { get; set; } = 1;

        protected virtual void Backward(Volume<T> y)
        {
            var chrono = Stopwatch.StartNew();

            var dimension = Ops<T>.Cast(y.Shape.GetDimension(3));
            this.Loss = Ops<T>.Divide(this.Net.Backward(y), dimension);
            this.BackwardTimeMs = chrono.Elapsed.TotalMilliseconds/y.Shape.GetDimension(3);
        }

        private void Forward(Volume<T> x)
        {
            var chrono = Stopwatch.StartNew();
            this.Net.Forward(x, true); // also set the flag that lets the net know we're just training
            this.ForwardTimeMs = chrono.Elapsed.TotalMilliseconds/x.Shape.GetDimension(3);
        }

        public void Train(Volume<T> x, Volume<T> y)
        {
            Forward(x);

            Backward(y);

            var chrono = Stopwatch.StartNew();
            TrainImplem();
            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds/x.Shape.GetDimension(3);
        }

        protected abstract void TrainImplem();
    }
}