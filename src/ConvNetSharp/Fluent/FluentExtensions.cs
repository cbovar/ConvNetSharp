using ConvNetSharp.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNetSharp.Fluent
{
    public static class FluentExtensions
    {
        #region LayerBase

        public static ReluLayer Relu(this LayerBase layer)
        {
            var relu = new ReluLayer();
            layer.ConnectTo(relu);

            return relu;
        }

        public static MaxoutLayer Maxout(this LayerBase layer)
        {
            var maxout = new MaxoutLayer();
            layer.ConnectTo(maxout);

            return maxout;
        }

        public static SigmoidLayer Sigmoid(this LayerBase layer)
        {
            var sigmoid = new SigmoidLayer();
            layer.ConnectTo(sigmoid);

            return sigmoid;
        }

        public static TanhLayer Tanh(this LayerBase layer)
        {
            var tanh = new TanhLayer();
            layer.ConnectTo(tanh);

            return tanh;
        }

        public static PoolLayer Pool(this LayerBase layer, int width, int height)
        {
            var pool = new PoolLayer(width, height);
            layer.ConnectTo(pool);

            return pool;
        }

        public static FullyConnLayer FullyConn(this LayerBase layer, int neuronCount)
        {
            var fullyConn = new FullyConnLayer(neuronCount);
            layer.ConnectTo(fullyConn);

            return fullyConn;
        }

        public static ConvLayer Conv(this LayerBase layer, int width, int height, int filterCount)
        {
            var conv = new ConvLayer(width, height, filterCount);
            layer.ConnectTo(conv);

            return conv;
        }

        public static SoftmaxLayer Softmax(this LayerBase layer, int classCount)
        {
            var softMax = new SoftmaxLayer(classCount);
            layer.ConnectTo(softMax);

            return softMax;
        }

        public static SvmLayer Svm(this LayerBase layer, int classCount)
        {
            var svm = new SvmLayer(classCount);
            layer.ConnectTo(svm);

            return svm;
        }

        public static RegressionLayer Regression(this LayerBase layer)
        {
            var regression = new RegressionLayer();
            layer.ConnectTo(regression);

            return regression;
        }

        public static DropOutLayer DropOut(this LayerBase layer, double prob = 0.5)
        {
            var dropout = new DropOutLayer(prob);
            layer.ConnectTo(dropout);

            return dropout;
        }

        public static FluentNet Build(this LastLayerBase layer)
        {
            return new FluentNet(layer);
        }

        #endregion

        #region ConvLayer

        public static ConvLayer Pad(this ConvLayer layer, int pad)
        {
            layer.Pad = pad;
            layer.UpdateOutputSize();
            return layer;
        }

        public static ConvLayer Stride(this ConvLayer layer, int stride)
        {
            layer.Stride = stride;
            layer.UpdateOutputSize();
            return layer;
        }

        public static ReluLayer Relu(this ConvLayer layer)
        {
            var relu = new ReluLayer();
            layer.ConnectTo(relu);

            layer.BiasPref = 0.1;
            layer.UpdateOutputSize();

            return relu;
        }

        #endregion

        #region PoolLayer

        public static PoolLayer Pad(this PoolLayer layer, int pad)
        {
            layer.Pad = pad;
            return layer;
        }

        public static PoolLayer Stride(this PoolLayer layer, int stride)
        {
            layer.Stride = stride;
            return layer;
        }

        #endregion
    }
}
