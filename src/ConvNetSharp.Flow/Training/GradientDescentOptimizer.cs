using System;
using System.Collections.Generic;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Training
{
    public class GradientDescentOptimizer<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Volume<T> _learningRate;
        private readonly Op<T> _updatedV;

        public GradientDescentOptimizer(T learningRate)
        {
            this._learningRate = learningRate;

            var lr = ConvNetSharp<T>.PlaceHolder("lr"); // learning rate
            var grad = ConvNetSharp<T>.PlaceHolder("grad"); // gradients
            var v = ConvNetSharp<T>.PlaceHolder("v"); // volume
            this._updatedV = v - grad * lr;
        }

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            var variables = session.LearnableVariables;

            foreach (var variable in variables.Values)
            {
                var variableV = session.Run(this._updatedV,
                    new Dictionary<string, Volume<T>>
                    {
                        {"lr", this._learningRate},
                        {"grad", variable.Derivate.Evaluate(session)},
                        {"v", variable.Evaluate(session)}
                    });
                variable.V = variableV.Clone();
            }

            return null;
        }

        public override string Representation => "Gradient Descent";
    }
}