using System;
using System.Diagnostics;

namespace ConvNetSharp.Training
{
    public abstract class TrainerBase
    {
        protected readonly INet Net;
        protected int K; // iteration counter

        protected TrainerBase(INet net)
        {
            this.Net = net;

            this.BatchSize = 1;
        }

        public TimeSpan BackwardTime { get; private set; }

        public double CostLoss { get; private set; }

        public TimeSpan ForwardTime { get; private set; }

        public int BatchSize { get; set; }

        public virtual double Loss
        {
            get { return this.CostLoss; }
        }

        public void Train(IVolume x, double y)
        {
            this.Forward(x);

            this.Backward(y);

            this.TrainImplem();
        }

        public void Train(double y, params IVolume[] x)
        {
            this.Forward(x);

            this.Backward(y);

            this.TrainImplem();
        }

        public void Train(IVolume x, double[] y)
        {
            this.Forward(x);

            this.Backward(y);

            this.TrainImplem();
        }

        public void Train(double[] y, params IVolume[] x)
        {
            this.Forward(x);

            this.Backward(y);

            this.TrainImplem();
        }

        protected abstract void TrainImplem();

        protected virtual void Backward(double y)
        {
            var chrono = Stopwatch.StartNew();
            this.CostLoss = this.Net.Backward(y);
            this.BackwardTime = chrono.Elapsed;
        }

        protected virtual void Backward(double[] y)
        {
            var chrono = Stopwatch.StartNew();
            this.CostLoss = this.Net.Backward(y);
            this.BackwardTime = chrono.Elapsed;
        }

        private void Forward(params IVolume[] x)
        {
            var chrono = Stopwatch.StartNew();
            this.Net.Forward(x, true); // also set the flag that lets the net know we're just training
            this.ForwardTime = chrono.Elapsed;
        }
    }
}