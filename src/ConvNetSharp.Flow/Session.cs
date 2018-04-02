using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ConvNetSharp.Flow.Graph;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow
{
    /// <summary>
    ///     TODO:
    ///     - scope management (to group ops together and to allow using the same name on different nodes)
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Session<T> : IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private readonly ConvNetSharp<T> _cns;
        private bool _derivativeComputed;

        public Session(ConvNetSharp<T> cns = null)
        {
            this._cns = cns ?? ConvNetSharp<T>.Instance;
        }

        public Op<T> Cost { get; private set; }

        public Dictionary<string, Variable<T>> LearnableVariables { get; set; } = new Dictionary<string, Variable<T>>();

        public long Step { get; set; }

        public int BatchSize { get; private set; }

        public bool IsTraining { get; set; } // Should this exist ?

        public void Dispose()
        {
            Op<T>.DisposeGraph(this.Cost);
        }

        /// <summary>
        ///     Automatic differentiation using reverse accumulation
        /// </summary>
        /// <param name="cost"></param>
        /// <param name="gradient">1 will be used as gradient if not specify</param>
        public void Differentiate(Op<T> cost, Op<T> gradient = null)
        {
            if (!this._derivativeComputed)
            {
                var visitor = new OpVisitor<T>(op => { op.Derivate = null; });
                cost.Accept(visitor);

                this.Cost = cost;

                cost.Derivate = gradient ?? ConvNetSharp<T>.One;

                var differentiateVisitor = new DifferentiateVisitor<T>();
                cost.Accept(differentiateVisitor);

                this._derivativeComputed = true;
            }
        }

        public void Dump(Op<T> fun, string fileName)
        {
            using (var sw = new StreamWriter(File.Create(fileName)))
            {
                var streamWriter = sw;
                var visitor = new OpVisitor<T>(op =>
                {
                    streamWriter.WriteLine(op);
                    streamWriter.WriteLine(op.Result == null ? "[Null]" : op.Result.ToString());
                });
                fun.Accept(visitor);
            }
        }

        public Op<T> GetVariableByName(Op<T> fun, string name)
        {
            Op<T> result = null;

            var visitor = new OpVisitor<T>(op =>
            {
                if (op is INamedOp<T> variable)
                {
                    if (variable.Name == name)
                    {
                        result = op;
                    }
                }
            });

            fun.Accept(visitor);

            return result;
        }

        /// <summary>
        /// Initialize place holders contained in the graph of the specify Op
        /// </summary>
        /// <param name="fun">Root of the graph to traverse</param>
        /// <param name="dictionary"></param>
        public void InitializePlaceHolders(Op<T> fun, Dictionary<string, Volume<T>> dictionary)
        {
            this.BatchSize = dictionary.Values.Select(o => o.Shape.GetDimension(3)).Max(); // is this correct?

            UpdatePlaceHolder(fun, dictionary);
        }

        public Volume<T> Run(Op<T> fun, Dictionary<string, Volume<T>> dictionary, bool incrementStep = true)
        {
            if (dictionary != null && dictionary.Any())
            {
                this.BatchSize = dictionary.Values.Select(o => o.Shape.GetDimension(3)).Max(); // is this correct?
                UpdatePlaceHolder(fun, dictionary);
            }

            var result = fun.Evaluate(this);

            if (incrementStep)
            {
                this.Step++;
            }

            return result;
        }

        public void UpdatePlaceHolder(Op<T> fun, Dictionary<string, Volume<T>> dictionary)
        {
            // Find all PlaceHolders and update their current value
            var visitor = new OpVisitor<T>(op =>
            {
                if (op is PlaceHolder<T> placeHolder)
                {
                    placeHolder.SetValue(dictionary[placeHolder.Name]);
                }

                if (op is Variable<T> variable && variable.IsLearnable)
                {
                    this.LearnableVariables[variable.Name] = variable;
                }
            });

            fun.Accept(visitor);
        }
    }
}