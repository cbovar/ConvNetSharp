using System;
using ConvNetSharp.Flow.Layers;

namespace ConvNetSharp.Flow.Fluent
{
    public static class FluentExtensions
    {
        #region LayerBase<T>

        public static DropoutLayer<T> DropoutLayer<T>(this LayerBase<T> layer, T dropoutProbability) where T : struct, IEquatable<T>, IFormattable
        {
            var dropout = new DropoutLayer<T>(dropoutProbability);
            dropout.AcceptParent(layer);

            return dropout;
        }

        public static ReluLayer<T> Relu<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var relu = new ReluLayer<T>();
            relu.AcceptParent(layer);

            return relu;
        }

        public static LeakyReluLayer<T> LeakyRelu<T>(this LayerBase<T> layer, T alpha) where T : struct, IEquatable<T>, IFormattable
        {
            var relu = new LeakyReluLayer<T>(alpha);
            relu.AcceptParent(layer);

            return relu;
        }

        public static SigmoidLayer<T> Sigmoid<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var sigmoid = new SigmoidLayer<T>();
            sigmoid.AcceptParent(sigmoid);

            return sigmoid;
        }

        public static TanhLayer<T> Tanh<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var tanh = new TanhLayer<T>();
            tanh.AcceptParent(layer);

            return tanh;
        }

        public static PoolLayer<T> Pool<T>(this LayerBase<T> layer, int width, int height) where T : struct, IEquatable<T>, IFormattable
        {
            var pool = new PoolLayer<T>(width, height);
            pool.AcceptParent(layer);

            return pool;
        }

        public static FullyConnLayer<T> FullyConn<T>(this LayerBase<T> layer, int neuronCount) where T : struct, IEquatable<T>, IFormattable
        {
            var fullyConn = new FullyConnLayer<T>(neuronCount);
            fullyConn.AcceptParent(layer);

            return fullyConn;
        }

        public static ConvLayer<T> Conv<T>(this LayerBase<T> layer, int width, int height, int filterCount) where T : struct, IEquatable<T>, IFormattable
        {
            var conv = new ConvLayer<T>(width, height, filterCount);
            conv.AcceptParent(layer);

            return conv;
        }

        public static SoftmaxLayer<T> Softmax<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var softMax = new SoftmaxLayer<T>();
            softMax.AcceptParent(layer);

            return softMax;
        }

        public static Net<T> Build<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var net = new Net<T>();
            net.AddLayer(layer);

            return net;
        }

        #endregion

        #region ConvLayer

        public static ConvLayer<T> Pad<T>(this ConvLayer<T> layer, int pad) where T : struct, IEquatable<T>, IFormattable
        {
            layer.Pad = pad;
            return layer;
        }

        public static ConvLayer<T> Stride<T>(this ConvLayer<T> layer, int stride) where T : struct, IEquatable<T>, IFormattable
        {
            layer.Stride = stride;
            return layer;
        }

        public static ReluLayer<T> Relu<T>(this ConvLayer<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var relu = new ReluLayer<T>();
            relu.AcceptParent(layer);

            layer.BiasPref = (T)Convert.ChangeType(0.1, typeof(T)); // can we do better?

            return relu;
        }

        public static LeakyReluLayer<T> LeakyRelu<T>(this ConvLayer<T> layer, T alpha) where T : struct, IEquatable<T>, IFormattable
        {
            var relu = new LeakyReluLayer<T>(alpha);
            relu.AcceptParent(layer);

            layer.BiasPref = (T)Convert.ChangeType(0.1, typeof(T)); // can we do better?

            return relu;
        }

        #endregion

        #region PoolLayer

        public static PoolLayer<T> Pad<T>(this PoolLayer<T> layer, int pad) where T : struct, IEquatable<T>, IFormattable
        {
            layer.Pad = pad;
            return layer;
        }

        public static PoolLayer<T> Stride<T>(this PoolLayer<T> layer, int stride) where T : struct, IEquatable<T>, IFormattable
        {
            layer.Stride = stride;
            return layer;
        }

        #endregion
    }
}