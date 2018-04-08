using System;

namespace ConvNetSharp.Flow.Training
{
    public class SgdTrainer<T> : TrainerBase<T>, IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        public SgdTrainer(Net<T> net, T learningRate) : base(net)
        {
            this.LearningRate = learningRate;
            this.Optimizer = new GradientDescentOptimizer<T>(net.Op.Graph, learningRate);
        }

        public T LearningRate { get; }

        public void Dispose()
        {
            this.Optimizer.Dispose();
        }
    }
}