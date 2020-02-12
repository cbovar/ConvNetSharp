using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Training
{
    public class AdamOptimizer<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly T _beta1;
        private readonly T _beta2;
        private readonly T _epsilon;
        private readonly Volume<T> _learningRate;
        private readonly Dictionary<Variable<T>, Op<T>> _updaters = new Dictionary<Variable<T>, Op<T>>();

        public AdamOptimizer(ConvNetSharp<T> graph, T learningRate, T beta1, T beta2, T epsilon) : base(graph)
        {
            this.LearningRate = learningRate;
            this._beta1 = beta1;
            this._beta2 = beta2;
            this._epsilon = epsilon;
            this._learningRate = BuilderInstance<T>.Volume.SameAs(new Shape(1));
        }

        public T LearningRate { get; set; }

        public override string Representation => "Adam";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this._learningRate?.Dispose();

                foreach (var op in this._updaters.Values)
                {
                    DisposeGraph(op);
                }
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            var variables = session.LearnableVariables;

            var volumes = new Dictionary<Variable<T>, Volume<T>>();
            var gradients = new Dictionary<Variable<T>, Volume<T>>();

            if (this._updaters.Count == 0)
            {
                foreach (var variable in variables.Values)
                {
                    var one = this.Graph.Const(Ops<T>.One, "one");
                    var epsilon = this.Graph.PlaceHolder("epsilon");
                    var beta1 = this.Graph.PlaceHolder("beta1");
                    var beta2 = this.Graph.PlaceHolder("beta2");
                    var m = this.Graph.Variable(Ops<T>.Zero, "m");
                    var v = this.Graph.Variable(Ops<T>.Zero, "v");
                    var t = this.Graph.Variable(Ops<T>.Zero, "t");
                    var grad = this.Graph.PlaceHolder("grad"); // gradients
                    var learningRate = this.Graph.PlaceHolder("lr"); // learning rate

                    var m_t = this.Graph.Assign(m, beta1 * m + (one - beta1) * grad); // m_t <- beta1 * m_{t-1} + (1 - beta1) * g
                    var v_t = this.Graph.Assign(v, beta2 * v + (one - beta2) * grad * grad); // beta2 * v_{t-1} + (1 - beta2) * g * g
                    var t_plus_1 = this.Graph.Assign(t, t + one); // t = t + 1
                    var lr = learningRate * this.Graph.Sqrt(one - (beta2 ^ t_plus_1)) / (one - (beta1 ^ t_plus_1)); // lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
                    var vol = this.Graph.PlaceHolder("vol");
                    var delta = m_t / (this.Graph.Sqrt(v_t) + epsilon);

                    this._updaters[variable] = vol - this.Graph.Sum(delta, this.Graph.Shape(vol)) * lr;
                }
            }

            this._learningRate.Set(0, Ops<T>.Divide(this.LearningRate, Ops<T>.Cast(session.BatchSize)));

            // Prepare updated variables
            foreach (var variable in variables.Values)
            {
                volumes[variable] = variable.Evaluate(session);
                gradients[variable] = variable.Derivate.Evaluate(session);
            }

            // Apply updated variables
            foreach (var variable in variables.Values)
            {
                var grad = gradients[variable];
                var v = volumes[variable];

                var variableV = session.Run(this._updaters[variable],
                    new Dictionary<string, Volume<T>>
                    {
                        {"epsilon", this._epsilon},
                        {"beta1", this._beta1},
                        {"beta2", this._beta2},
                        {"lr", this._learningRate},
                        {"grad", grad},
                        {"vol", v}
                    }, false);

                variable.Result.Storage.CopyFrom(variableV.Storage);
                variable.SetDirty();
            }

            return null;
        }
    }
}