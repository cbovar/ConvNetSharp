using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow
{
    public class ConvNetSharp<T> where T : struct, IEquatable<T>, IFormattable
    {
        public static readonly T Zero;

        public static readonly T One;

        private static readonly Lazy<ConvNetSharp<T>> Lazy = new Lazy<ConvNetSharp<T>>(() => new ConvNetSharp<T>());

        private readonly Stack<string> _scopes = new Stack<string>();

        static ConvNetSharp()
        {
            Zero = default(T);

            if (typeof(T) == typeof(double))
            {
                One = (T)(ValueType)1.0;
            }
            else if (typeof(T) == typeof(float))
            {
                One = (T)(ValueType)1.0f;
            }
        }

        public static ConvNetSharp<T> Instance => Lazy.Value;

        public Const<T> Const(Volume<T> v, string name)
        {
            return new Const<T>(v, name);
        }

        public Convolution<T> Conv(Op<T> x, int width, int height, int filterCount, int stride = 1, int pad = 0)
        {
            return new Convolution<T>(x, width, height, filterCount, stride, pad);
        }

        public Op<T> CrossEntropyLoss(Op<T> x, Op<T> y)
        {
            return new SoftmaxCrossEntropy<T>(x, y);
        }

        public Op<T> Exp(Op<T> x)
        {
            return new Exp<T>(x);
        }

        public Op<T> Log(Op<T> x)
        {
            return new Log<T>(x);
        }

        public PlaceHolder<T> PlaceHolder(string name)
        {
            return new PlaceHolder<T>(name);
        }

        public Pool<T> Pool(Op<T> x, int width, int height, int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            return new Pool<T>(x, width, height, horizontalPad, verticalPad, horizontalStride, verticalStride);
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
            return new Activation<T>(x, ActivationType.Relu);
        }

        public Op<T> Reshape(Op<T> x, Shape shape)
        {
            return new Reshape<T>(x, shape);
        }

        public Op<T> Reshape(Op<T> x, Op<T> shape)
        {
            return new Reshape<T>(x, shape);
        }

        public Op<T> Flatten(Op<T> x)
        {
            return Reshape(x, new Shape(1, 1, -1, Volume.Shape.Keep));
        }

        public Scope<T> Scope(string name)
        {
            RegisterScope(name);
            return new Scope<T>(name, this);
        }

        public Op<T> Shape(Op<T> x)
        {
            return new Shape<T>(x);
        }

        public Op<T> Sigmoid(Op<T> x)
        {
            return new Activation<T>(x, ActivationType.Sigmoid);
        }

        public Op<T> Softmax(Op<T> x)
        {
            return new Softmax<T>(x);
        }

        public Op<T> Sum(Op<T> x, Shape shape)
        {
            return new Sum<T>(x, shape);
        }

        public Op<T> Sum(Op<T> x, Op<T> shape)
        {
            return new Sum<T>(x, shape);
        }

        public Op<T> Tanh(Op<T> x)
        {
            return new Activation<T>(x, ActivationType.Tanh);
        }

        public Variable<T> Variable(Volume<T> v, string name)
        {
            var agg = this._scopes.Reverse().Aggregate("", (s1, s2) => s1 + s2 + "/");
            return new Variable<T>(v, agg + name);
        }

        public Variable<T> Variable(Shape shape, string name)
        {
            return this.Variable(BuilderInstance<T>.Volume.SameAs(shape), name);
        }

        public Variable<T> Variable(string name)
        {
            var agg = this._scopes.Reverse().Aggregate("", (s1, s2) => s1 + s2 + "/");
            return new Variable<T>(null, agg + name);
        }
    }
}