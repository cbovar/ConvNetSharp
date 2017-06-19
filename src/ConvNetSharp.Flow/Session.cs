using System;
using System.Collections.Generic;
using System.IO;
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

        public void Dispose()
        {
            var visitor = new OpVisitor<T>(op => { op.Dispose(); });
            this.Cost?.Accept(visitor);
        }

        /// <summary>
        ///     Automatic differentiation using reverse accumulation
        /// </summary>
        /// <param name="cost"></param>
        public void Differentiate(Op<T> cost)
        {
            if (!this._derivativeComputed)
            {
                this.Cost = cost;

                cost.Derivate = this._cns.Const(ConvNetSharp<T>.One, "1");

                //this._func.Derivate = cost;
                var differentiateVisitor = new DifferentiateVisitor<T>();
                cost.Accept(differentiateVisitor);

                this._derivativeComputed = true;
            }
        }

        public Volume<T> Run(Op<T> fun, Dictionary<string, Volume<T>> dictionary)
        {
            // Find all PlaceHolders and update their current value
            var visitor = new OpVisitor<T>(op =>
            {
                var placeHolder = op as PlaceHolder<T>;
                if (placeHolder != null)
                {
                    placeHolder.Result = dictionary[placeHolder.Name];
                    placeHolder.SetDirty();
                }

                var variable = op as Variable<T>;
                if (variable != null)
                {
                    this.LearnableVariables[variable.Name] = variable;
                }
            });
            fun.Accept(visitor);

            var result = fun.Evaluate(this);

            this.Step++;

            return result;
        }

        public void Dump(Op<T> fun, string fileName)
        {
            using (var sw = new StreamWriter(File.Create(fileName)))
            {
                var visitor = new OpVisitor<T>(op =>
                {
//                    var variable = op as Variable<T>;
                    //if (variable != null)
                    {
                        sw.WriteLine(op);
                        sw.WriteLine(op.Result == null ? "[Null]" : op.Result.ToString());
                    }
                });
                fun.Accept(visitor);
            }
        }
    }
}