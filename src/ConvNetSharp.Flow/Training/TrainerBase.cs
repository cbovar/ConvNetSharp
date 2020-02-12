using System;
using System.Collections.Generic;
using System.Diagnostics;
using ConvNetSharp.Core;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Training
{
    public abstract class TrainerBase<T> : Core.Training.TrainerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Dictionary<string, Volume<T>> _dico = new Dictionary<string, Volume<T>>();
        protected readonly Net<T> _net;

        protected TrainerBase(INet<T> net) : base(net)
        {
            this._net = net as Net<T>;
        }

        public Op<T> Optimizer { get; set; }

        protected override void TrainImplem()
        {
            throw new NotImplementedException();
        }

        public override void Train(Volume<T> x, Volume<T> y)
        {
            var batchSize = x.Shape.Dimensions[3];

            this._dico["Y"] = y;
            this._dico["input"] = x;

            var chrono = Stopwatch.StartNew();
            this.Loss = this._net.Session.Run(this._net.Cost, this._dico).Get(0);
            this.ForwardTimeMs = chrono.Elapsed.TotalMilliseconds / batchSize;

            chrono = Stopwatch.StartNew();
            this._net.Session.Run(this.Optimizer, this._dico);
            this.BackwardTimeMs = chrono.Elapsed.TotalMilliseconds / batchSize;
        }
    }
}