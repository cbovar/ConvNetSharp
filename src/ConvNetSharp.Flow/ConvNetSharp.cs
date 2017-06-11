using System;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow
{
    public static class ConvNetSharp<T> where T : struct, IEquatable<T>, IFormattable
    {
        public static readonly T Zero;

        public static readonly T One;

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

        public static Const<T> Const(Volume<T> v, string name)
        {
            return new Const<T>(v, name);
        }

        public static PlaceHolder<T> PlaceHolder(string name)
        {
            return new PlaceHolder<T>(name);
        }

        public static Variable<T> Variable(Volume<T> v, string name)
        {
            return new Variable<T>(v, name);
        }

        public static Op<T> Sigmoid(Op<T> x)
        {
            return new Activation<T>(x, ActivationType.Sigmoid);
        }

        public static Op<T> Softmax(Op<T> x)
        {
            return new SoftmaxOp<T>(x);
        }

        public static Op<T> CrossEntropyLoss(Op<T> x, Op<T> y)
        {
            return new CrossEntropyLoss<T>(x, y);
        }

        public static Op<T> Relu(Op<T> x)
        {
            return new Activation<T>(x, ActivationType.Relu);
        }

        public static Op<T> Tanh(Op<T> x)
        {
            return new Activation<T>(x, ActivationType.Tanh);
        }

        public static Op<T> Log(Op<T> x)
        {
            return new Log<T>(x);
        }

        public static Op<T> Exp(Op<T> x)
        {
            return new Exp<T>(x);
        }

        public static Op<T> Conv(Op<T> x, int width, int height, int filterCount, int stride = 1, int pad = 0)
        {
            return new Convolution<T>(x, width, height, filterCount, stride, pad);
        }
    }
}