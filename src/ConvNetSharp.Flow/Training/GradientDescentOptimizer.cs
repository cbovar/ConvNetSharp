using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Training
{
    public class GradientDescentOptimizer<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly ConvNetSharp<T> _cns;
        private readonly Volume<T> _learningRate;
        private readonly T _lr;
        private readonly Dictionary<Variable<T>, Volume<T>> _tempGrads = new Dictionary<Variable<T>, Volume<T>>();
        private readonly Dictionary<Variable<T>, Op<T>> _updaters = new Dictionary<Variable<T>, Op<T>>();

        public GradientDescentOptimizer(T learningRate, ConvNetSharp<T> cns = null)
        {
            this._lr = learningRate;
            this._learningRate = BuilderInstance<T>.Volume.SameAs(new Shape(1));
            this._cns = cns ?? ConvNetSharp<T>.Instance;
        }

        public override string Representation => "Gradient Descent";

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

                foreach (var vol in this._tempGrads.Values)
                {
                    vol.Dispose();
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
                    var lr = this._cns.PlaceHolder("lr"); // learning rate
                    var grad = this._cns.PlaceHolder("grad"); // gradients
                    var v = this._cns.PlaceHolder("v"); // volume

                    this._updaters[variable] = v - grad * lr;
                }
            }

            this._learningRate.Set(0, Ops<T>.Divide(this._lr, Ops<T>.Cast(session.BatchSize)));

            // Prepare updated variables
            foreach (var variable in variables.Values)
            {
                var grad = variable.Derivate.Evaluate(session);
                var volume = variable.Evaluate(session);

                var gradBatchSize = grad.Shape.GetDimension(3);
                var volumeBatchSize = volume.Shape.GetDimension(3);

                if (gradBatchSize != volumeBatchSize && gradBatchSize != 1)
                {
                    // Batch size > 1

                    var gradShape = new Shape(grad.Shape);
                    gradShape.SetDimension(0, variable.Result.Shape.GetDimension(0));
                    gradShape.SetDimension(1, variable.Result.Shape.GetDimension(1));
                    gradShape.SetDimension(3, 1);

                    Volume<T> tempGrad;
                    if (!this._tempGrads.TryGetValue(variable, out tempGrad) || !tempGrad.Shape.Equals(gradShape))
                    {
                        tempGrad = BuilderInstance<T>.Volume.SameAs(gradShape);
                        this._tempGrads[variable] = tempGrad;
                    }

                    grad.DoSum(tempGrad); // sum gradient batch
                    grad = tempGrad;
                }

                volumes[variable] = volume;
                gradients[variable] = grad;
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