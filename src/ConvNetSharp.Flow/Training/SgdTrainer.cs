using System;
using ConvNetSharp.Core;

namespace ConvNetSharp.Flow.Training
{
    public class SgdTrainer<T> : TrainerBase<T>, IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        public SgdTrainer(INet<T> net, T learningRate) : base(net)
        {
            this.LearningRate = learningRate;
            this.Optimizer = new GradientDescentOptimizer<T>(learningRate);
        }

        public T LearningRate { get; }

        public void Dispose()
        {
            this.Optimizer.Dispose();
        }
    }
}