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
        private readonly T _lr;

        private Volume<T> _tempGrad;

        public GradientDescentOptimizer(T learningRate, ConvNetSharp<T> cns = null)
        {
            this._lr = learningRate;
            this._learningRate = BuilderInstance<T>.Volume.SameAs(new Shape(1));
            cns = cns ?? ConvNetSharp<T>.Instance;

            var lr = cns.PlaceHolder("lr"); // learning rate
            var grad = cns.PlaceHolder("grad"); // gradients
            var v = cns.PlaceHolder("v"); // volume

            this.UpdatedV = -grad * lr + v;
        }

        public Op<T> UpdatedV { get; }

        public override string Representation => "Gradient Descent";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            var variables = session.LearnableVariables;

            var dico = new Dictionary<Variable<T>, Volume<T>>();

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
                    gradShape.SetDimension(3, 1);

                    if (this._tempGrad == null || !this._tempGrad.Shape.Equals(gradShape))
                    {
                        this._tempGrad = BuilderInstance<T>.Volume.SameAs(gradShape);
                    }

                    grad.DoSum(this._tempGrad); // sum gradient over all batches 
                    grad = this._tempGrad;
                }

#if DEBUG
                //  Console.WriteLine(variable);
                var inputs = grad.ToArray();
                foreach (var i in inputs)
                {
                    // Console.WriteLine(i);
                    if (Ops<T>.IsInvalid(i))
                    {
                        throw new ArgumentException("Invalid input!");
                    }
                }
                //  Console.WriteLine(Environment.NewLine);
#endif

                var dimension = volume.Shape.GetDimension(3);
                this._learningRate.Set(0, Ops<T>.Divide(this._lr, Ops<T>.Cast(13)));

                var variableV = session.Run(this.UpdatedV,
                    new Dictionary<string, Volume<T>>
                    {
                        {"lr", this._learningRate},
                        {"grad", grad},
                        {"v", volume}
                    });

                dico[variable] = variableV;
            }

            // Apply updated variables
            foreach (var pair in dico)
            {
                pair.Key.Result.Storage.CopyFrom(pair.Value.Storage);
            }

            return null;
        }
    }
}