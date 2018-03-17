using System;
using ConvNetSharp.Core.Layers;

namespace ConvNetSharp.Core.Fluent
{
    public static class FluentExtensions
    {
        #region LayerBase<T>

        public static DropoutLayer<T> DropoutLayer<T>(this LayerBase<T> layer, T dropProbability) where T : struct, IEquatable<T>, IFormattable
        {
            var pool = new DropoutLayer<T>(dropProbability);
            layer.ConnectTo(pool);

            return pool;
        }

        public static ReluLayer<T> Relu<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var relu = new ReluLayer<T>();
            layer.ConnectTo(relu);

            return relu;
        }

        public static SigmoidLayer<T> Sigmoid<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var sigmoid = new SigmoidLayer<T>();
            layer.ConnectTo(sigmoid);

            return sigmoid;
        }

        public static TanhLayer<T> Tanh<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var tanh = new TanhLayer<T>();
            layer.ConnectTo(tanh);

            return tanh;
        }

        public static PoolLayer<T> Pool<T>(this LayerBase<T> layer, int width, int height) where T : struct, IEquatable<T>, IFormattable
        {
            var pool = new PoolLayer<T>(width, height);
            layer.ConnectTo(pool);

            return pool;
        }

        public static FullyConnLayer<T> FullyConn<T>(this LayerBase<T> layer, int neuronCount) where T : struct, IEquatable<T>, IFormattable
        {
            var fullyConn = new FullyConnLayer<T>(neuronCount);
            layer.ConnectTo(fullyConn);

            return fullyConn;
        }

        public static ConvLayer<T> Conv<T>(this LayerBase<T> layer, int width, int height, int filterCount) where T : struct, IEquatable<T>, IFormattable
        {
            var conv = new ConvLayer<T>(width, height, filterCount);
            layer.ConnectTo(conv);

            return conv;
        }

        public static SoftmaxLayer<T> Softmax<T>(this LayerBase<T> layer, int classCount) where T : struct, IEquatable<T>, IFormattable
        {
            var softMax = new SoftmaxLayer<T>(classCount);
            layer.ConnectTo(softMax);

            return softMax;
        }

        //public static SvmLayer<T> Svm<T>(this LayerBase<T> layer, int classCount) where T : struct, IEquatable<T>, IFormattable
        //{
        //    var svm = new SvmLayer<T>(classCount);
        //    layer.ConnectTo(svm);

        //    return svm;
        //}

        public static RegressionLayer<T> Regression<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var regression = new RegressionLayer<T>();
            layer.ConnectTo(regression);

            return regression;
        }

        public static FluentNet<T> Build<T>(this LastLayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            return new FluentNet<T>(layer);
        }

        #endregion

        #region ConvLayer

        public static ConvLayer<T> Pad<T>(this ConvLayer<T> layer, int pad) where T : struct, IEquatable<T>, IFormattable
        {
            layer.Pad = pad;
            layer.UpdateOutputSize();
            return layer;
        }

        public static ConvLayer<T> Stride<T>(this ConvLayer<T> layer, int stride) where T : struct, IEquatable<T>, IFormattable
        {
            layer.Stride = stride;
            layer.UpdateOutputSize();
            return layer;
        }

        public static ReluLayer<T> Relu<T>(this ConvLayer<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var relu = new ReluLayer<T>();
            layer.ConnectTo(relu);

            layer.BiasPref = (T)Convert.ChangeType(0.1, typeof(T)); // can we do better?
            layer.UpdateOutputSize();

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