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

        public T CostLoss { get; private set; }

        public double ForwardTimeMs { get; private set; }

        public virtual T Loss => this.CostLoss;

        public int BatchSize { get; set; } = 1;

        protected virtual void Backward(Volume<T> y)
        {
            var chrono = Stopwatch.StartNew();

            if (typeof(T) == typeof(double))
            {
                this.CostLoss = (T) (object) ((double) (object) this.Net.Backward(y) / y.Shape.GetDimension(3));
            }
            else if (typeof(T) == typeof(float))
            {
                this.CostLoss = (T) (object) ((float) (object) this.Net.Backward(y) / y.Shape.GetDimension(3));
            }

            this.BackwardTimeMs = chrono.Elapsed.TotalMilliseconds / y.Shape.GetDimension(3);
        }

        private void Forward(Volume<T> x)
        {
            var chrono = Stopwatch.StartNew();
            this.Net.Forward(x, true); // also set the flag that lets the net know we're just training
            this.ForwardTimeMs = chrono.Elapsed.TotalMilliseconds / x.Shape.GetDimension(3);
        }

        public void Train(Volume<T> x, Volume<T> y)
        {
            Forward(x);

            Backward(y);

            TrainImplem();
        }

        protected abstract void TrainImplem();
    }
}