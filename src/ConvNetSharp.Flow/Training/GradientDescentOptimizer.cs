using System;
using System.Collections.Generic;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using ConvNetSharp.Core;

namespace ConvNetSharp.Flow.Training
{
    public class GradientDescentOptimizer<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Volume<T> _learningRate;
        private readonly Op<T> _updatedV;

        public GradientDescentOptimizer(T learningRate, ConvNetSharp<T> cns = null)
        {
            this._learningRate = learningRate;
            cns = cns ?? ConvNetSharp<T>.Instance;

            var lr = cns.PlaceHolder("lr"); // learning rate
            var grad = cns.PlaceHolder("grad"); // gradients
            var v = cns.PlaceHolder("v"); // volume
            this._updatedV = v - grad * lr;
        }

        public override string Representation => "Gradient Descent";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            var variables = session.LearnableVariables;

            foreach (var variable in variables.Values)
            {
                var grad = variable.Derivate.Evaluate(session);

#if DEBUG
                Console.WriteLine(variable);
                var inputs = grad.ToArray();
                foreach (var i in inputs)
                {
                    Console.WriteLine(i);
                    if (Ops<T>.IsInvalid(i))
                        throw new ArgumentException("Invalid input!");
                }
#endif

                var variableV = session.Run(this._updatedV,
                    new Dictionary<string, Volume<T>>
                    {
                        {"lr", this._learningRate},
                        {"grad", grad},
                        {"v", variable.Evaluate(session)}
                    });

                variable.V.Storage.CopyFrom(variableV.Storage);
            }

            Console.WriteLine("------");

            return null;
        }
    }
}