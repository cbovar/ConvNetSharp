using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Training
{
    public class GradientDescentOptimizer<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Volume<T> _learningRate;
        private readonly Dictionary<Variable<T>, Op<T>> _updaters = new Dictionary<Variable<T>, Op<T>>();

        public GradientDescentOptimizer(ConvNetSharp<T> graph, T learningRate, ConvNetSharp<T> cns = null) : base(graph)
        {
            this.LearningRate = learningRate;
            this._learningRate = BuilderInstance<T>.Volume.SameAs(new Shape(1));
        }

        public T LearningRate { get; set; }

        public override string Representation => "Sgd";

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
                    var lr = this.Graph.PlaceHolder("lr"); // learning rate
                    var grad = this.Graph.PlaceHolder("grad"); // gradients
                    var v = this.Graph.PlaceHolder("v"); // volume

                    this._updaters[variable] = v - this.Graph.Sum(grad, this.Graph.Shape(v)) * lr;
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
                        {"lr", this._learningRate},
                        {"grad", grad},
                        {"v", v}
                    }, false);

                variable.Result.Storage.CopyFrom(variableV.Storage);
                variable.SetDirty();
            }

            return null;
        }
    }
}