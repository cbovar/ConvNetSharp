using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Training
{
    public class AdamOptimizer<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly ConvNetSharp<T> _cns;
        private readonly Volume<T> _learningRate;
        private readonly T _lr;
        private readonly T _beta1;
        private readonly T _beta2;
        private readonly T _epsilon;
        private readonly Dictionary<Variable<T>, Volume<T>> _tempGrads = new Dictionary<Variable<T>, Volume<T>>();
        private readonly Dictionary<Variable<T>, Op<T>> _updaters = new Dictionary<Variable<T>, Op<T>>();

        public AdamOptimizer(T learningRate, T beta1, T beta2, T epsilon, ConvNetSharp<T> cns = null)
        {
            this._lr = learningRate;
            this._beta1 = beta1;
            this._beta2 = beta2;
            this._epsilon = epsilon;
            this._learningRate = BuilderInstance<T>.Volume.SameAs(new Shape(1));
            this._cns = cns ?? ConvNetSharp<T>.Instance;
        }

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
                    var one = this._cns.Const(Ops<T>.One, "one");
                    var epsilon = this._cns.PlaceHolder("epsilon");
                    var beta1 = this._cns.PlaceHolder("beta1");
                    var beta2 = this._cns.PlaceHolder("beta2");
                    var m = this._cns.Variable(Ops<T>.Zero, "m");
                    var v = this._cns.Variable(Ops<T>.Zero, "v");
                    var t = this._cns.Variable(Ops<T>.Zero, "t");
                    var grad = this._cns.PlaceHolder("grad"); // gradients
                    var learningRate = this._cns.PlaceHolder("lr"); // learning rate

                    var m_t = this._cns.Assign(m, beta1 * m + (one - beta1) * grad);  // m_t <- beta1 * m_{t-1} + (1 - beta1) * g
                                                                                      //  m_t.Evaluated += (sender, args) => { Console.WriteLine($"m[{variable}]={ ((Op<T>)sender).Result.Get(0)}"); };

                    var v_t = this._cns.Assign(v, beta2 * v + (one - beta2) * grad * grad);  // beta2 * v_{t-1} + (1 - beta2) * g * g
                    //v_t.Evaluated += (sender, args) => { Console.WriteLine($"v[{variable}]={ ((Op<T>)sender).Result.Get(0)}"); };

                    var t_plus_1 = this._cns.Assign(t, t + one); // t = t + 1
                    //t_plus_1.Evaluated += (sender, args) => { Console.WriteLine($"t[{variable}]={ ((Op<T>)sender).Result.Get(0)}"); };

                    var lr = learningRate * this._cns.Sqrt(one - (beta2 ^ t_plus_1)) / (one - (beta1 ^ t_plus_1)); // lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
                    //lr.Evaluated += (sender, args) => { Console.WriteLine($"lr[{variable}]={ ((Op<T>)sender).Result.Get(0)}"); };

                    var vol = this._cns.PlaceHolder("vol");

                    var delta = lr * (m_t / (this._cns.Sqrt(v_t) + epsilon));
                    //delta.Evaluated += (sender, args) => { Console.WriteLine($"delta[{variable}]={ ((Op<T>)sender).Result.Get(0)}"); };

                    this._updaters[variable] = vol - delta;
                }
            }

            this._learningRate.Set(0, Ops<T>.Divide(this._lr, Ops<T>.Cast(session.BatchSize)));

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
                        {"epsilon", this._epsilon },
                        {"beta1", this._beta1 },
                        {"beta2", this._beta2 },
                        {"lr", this._learningRate},
                        {"grad", grad},
                        {"vol", v}
                    }, false);

                variable.Result.Storage.CopyFrom(variableV.Storage);
                variable.SetDirty();

                //    Console.WriteLine("-----------------");
            }

            return null;
        }
    }
}