using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow
{
    /// <summary>
    ///     Class containing convenience methods to build a computation graph
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ConvNetSharp<T> where T : struct, IEquatable<T>, IFormattable
    {
        public static readonly T Zero;
        public static readonly T One;

        private static readonly Lazy<ConvNetSharp<T>> Lazy = new Lazy<ConvNetSharp<T>>(() => new ConvNetSharp<T>());
        private static ConvNetSharp<T> _default;

        private readonly Stack<string> _scopes = new Stack<string>();

        static ConvNetSharp()
        {
            Zero = default;

            if (typeof(T) == typeof(double))
            {
                One = (T)(ValueType)1.0;
            }
            else if (typeof(T) == typeof(float))
            {
                One = (T)(ValueType)1.0f;
            }
        }

        public ConvNetSharp(bool setDefault = true)
        {
            if (setDefault)
            {
                Default = this;
            }
        }

        public static ConvNetSharp<T> Default
        {
            get => _default ?? Lazy.Value;
            set => _default = value;
        }

        public Op<T> Assign(Op<T> valueOp, Op<T> op)
        {
            return new Assign<T>(this, valueOp, op);
        }

        public Op<T> Concat(Op<T> x, Op<T> y)
        {
            return new Concat<T>(this, x, y);
        }

        public Const<T> Const(Volume<T> v, string name)
        {
            return new Const<T>(this, v, name);
        }

        public Const<T> Const(T x, string name)
        {
            return new Const<T>(this, x, name);
        }

        public Convolution<T> Conv(Op<T> x, int width, int height, int filterCount, int stride = 1, int pad = 0)
        {
            return new Convolution<T>(this, x, width, height, filterCount, stride, pad);
        }

        public Op<T> CrossEntropyLoss(Op<T> x, Op<T> y)
        {
            return new SoftmaxCrossEntropy<T>(this, x, y);
        }

        public Dense<T> Dense(Op<T> x, int neuronCount)
        {
            return new Dense<T>(this, x, neuronCount);
        }

        public Op<T> Dropout(Op<T> x, Op<T> dropoutProbability)
        {
            return new Dropout<T>(this, x, dropoutProbability);
        }

        public Op<T> Exp(Op<T> x)
        {
            return new Exp<T>(this, x);
        }

        public Op<T> Extract(Op<T> x, Op<T> length, Op<T> offset)
        {
            return new Extract<T>(this, x, length, offset);
        }

        public Op<T> Flatten(Op<T> x)
        {
            return this.Reshape(x, new Shape(1, 1, -1, Volume.Shape.Keep));
        }

        public Op<T> LeakyRelu(Op<T> x, T alpha)
        {
            return new LeakyRelu<T>(this, x, alpha);
        }

        public Op<T> LeakyReluGradient(Op<T> y, Op<T> derivate, T alpha)
        {
            return new LeakyReluGradient<T>(this, y, derivate, alpha);
        }

        public Op<T> Log(Op<T> x)
        {
            return new Log<T>(this, x);
        }

        public Op<T> MatMult(Op<T> x, Op<T> y)
        {
            return new MatMult<T>(this, x, y);
        }

        public Op<T> Max(Op<T> x)
        {
            return new Max<T>(this, x);
        }

        public Op<T> Negate(Op<T> x)
        {
            return new Negate<T>(this, x);
        }

        public PlaceHolder<T> PlaceHolder(string name)
        {
            return new PlaceHolder<T>(this, name);
        }

        public Pool<T> Pool(Op<T> x, int width, int height, int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            return new Pool<T>(this, x, width, height, horizontalPad, verticalPad, horizontalStride, verticalStride);
        }

        public Op<T> Power(Op<T> u, Op<T> v)
        {
            return new Power<T>(this, u, v);
        }

        public void RegisterScope(string name)
        {
            this._scopes.Push(name);
        }

        public void ReleaseScope(string name)
        {
            var popped = this._scopes.Pop();

            if (popped != name)
            {
                throw new ArgumentException($"Released scope should be '{popped} 'but was '{name}'");
            }
        }

        public Op<T> Relu(Op<T> x)
        {
            return new Activation<T>(this, x, ActivationType.Relu);
        }

        public Op<T> Reshape(Op<T> x, Shape shape)
        {
            return new Reshape<T>(this, x, shape);
        }

        public Op<T> Reshape(Op<T> x, Op<T> shape)
        {
            return new Reshape<T>(this, x, shape);
        }

        public Scope<T> Scope(string name)
        {
            this.RegisterScope(name);
            return new Scope<T>(name, this);
        }

        public Op<T> Shape(Op<T> x)
        {
            return new Shape<T>(this, x);
        }

        public Op<T> Sigmoid(Op<T> x)
        {
            return new Activation<T>(this, x, ActivationType.Sigmoid);
        }

        public Op<T> Softmax(Op<T> x)
        {
            return new Softmax<T>(this, x);
        }

        public Op<T> Sqrt(Op<T> x)
        {
            return new Sqrt<T>(this, x);
        }

        public Op<T> Sum(Op<T> x, Shape shape)
        {
            return new Sum<T>(this, x, shape);
        }

        public Op<T> Sum(Op<T> x, Op<T> shape)
        {
            return new Sum<T>(this, x, shape);
        }

        public Op<T> Tanh(Op<T> x)
        {
            return new Activation<T>(this, x, ActivationType.Tanh);
        }

        public Op<T> Tile(Op<T> x, Op<T> reps)
        {
            return new Tile<T>(this, x, reps);
        }

        public Op<T> Transpose(Op<T> x)
        {
            return new Transpose<T>(this, x);
        }

        public Variable<T> Variable(Volume<T> v, string name, bool isLearnable = false)
        {
            var agg = this._scopes.Reverse().Aggregate("", (s1, s2) => s1 + s2 + "/");
            return new Variable<T>(this, v, agg + name, isLearnable);
        }

        public Variable<T> Variable(Shape shape, string name, bool isLearnable = false)
        {
            return this.Variable(BuilderInstance<T>.Volume.SameAs(shape), name, isLearnable);
        }

        public Variable<T> Variable(string name, bool isLearnable = false)
        {
            var agg = this._scopes.Reverse().Aggregate("", (s1, s2) => s1 + s2 + "/");
            return new Variable<T>(this, null, agg + name, isLearnable);
        }
    }
}