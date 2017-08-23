using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Serialization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    /// <summary>
    /// TODO: make it generic
    /// </summary>
    [TestClass]
    public class SerializationTests
    {
        [TestMethod]
        public void ConvLayerSerialization()
        {
            var layer = new ConvLayer(5, 5, 8) { Stride = 1, Pad = 2, BiasPref = 0.5 };
            layer.Init(28, 24, 1);

            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as ConvLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);

            Assert.AreEqual(layer.Width, deserialized.Width);
            Assert.AreEqual(layer.Height, deserialized.Height);
            Assert.AreEqual(layer.Pad, deserialized.Pad);
            Assert.AreEqual(layer.Stride, deserialized.Stride);
            Assert.AreEqual(layer.FilterCount, deserialized.FilterCount);

            Assert.AreEqual(layer.Filters.Shape, deserialized.Filters.Shape);
            Assert.IsTrue(layer.Filters.ToArray().SequenceEqual(deserialized.Filters.ToArray()));

            Assert.AreEqual(layer.Bias.Shape, deserialized.Bias.Shape);
            Assert.IsTrue(layer.Bias.ToArray().SequenceEqual(deserialized.Bias.ToArray()));

            Assert.AreEqual(layer.BiasPref, deserialized.BiasPref);
        }

        [TestMethod]
        public void FullyConnLayerSerialization()
        {
            var layer = new FullyConnLayer(10) { BiasPref = 0.5 };
            layer.Init(28, 24, 1);

            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as FullyConnLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);

            Assert.AreEqual(layer.NeuronCount, deserialized.NeuronCount);

            Assert.AreEqual(layer.Filters.Shape, deserialized.Filters.Shape);
            Assert.IsTrue(layer.Filters.ToArray().SequenceEqual(deserialized.Filters.ToArray()));

            Assert.AreEqual(layer.Bias.Shape, deserialized.Bias.Shape);
            Assert.IsTrue(layer.Bias.ToArray().SequenceEqual(deserialized.Bias.ToArray()));

            Assert.AreEqual(layer.BiasPref, deserialized.BiasPref);
        }

        [TestMethod]
        public void InputLayerSerialization()
        {
            var layer = new InputLayer(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as InputLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
        }

        [TestMethod]
        public void NetSerialization()
        {
            var net = new Net<double>();
            net.AddLayer(new InputLayer(28, 28, 1));
            net.AddLayer(new ConvLayer(5, 5, 8) { Stride = 1, Pad = 2, BiasPref = 0.1 });
            net.AddLayer(new ReluLayer());
            net.AddLayer(new PoolLayer(2, 2) { Stride = 2 });
            net.AddLayer(new SigmoidLayer());
            net.AddLayer(new TanhLayer());
            net.AddLayer(new FullyConnLayer(10) { BiasPref = 0.2 });
            net.AddLayer(new SoftmaxLayer(10));

            var json = net.ToJson();
            var deserialized = SerializationExtensions.FromJson<double>(json);
        }

        [TestMethod]
        public void NetSerializationData()
        {
            var net = new Net<double>();
            net.AddLayer(new InputLayer(28, 28, 1));
            net.AddLayer(new ConvLayer(5, 5, 8) { Stride = 1, Pad = 2 });
            net.AddLayer(new ReluLayer());
            net.AddLayer(new PoolLayer(2, 2) { Stride = 2 });
            net.AddLayer(new ConvLayer(5, 5, 16) { Stride = 1, Pad = 2 });
            net.AddLayer(new ReluLayer());
            net.AddLayer(new PoolLayer(3, 3) { Stride = 3 });
            net.AddLayer(new FullyConnLayer(10));
            net.AddLayer(new SoftmaxLayer(10));

            var data = net.GetData();

            var layers = data["Layers"] as List<Dictionary<string, object>>;
            Assert.IsNotNull(layers);
            Assert.AreEqual(net.Layers.Count, layers.Count);

            var deserialized = Net<double>.FromData(data);
            Assert.AreEqual(net.Layers.Count, deserialized.Layers.Count);
        }

        [TestMethod]
        public void PoolLayerSerialization()
        {
            var layer = new PoolLayer(3, 3) { Pad = 1, Stride = 2 };
            layer.Init(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as PoolLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);

            Assert.AreEqual(layer.Width, deserialized.Width);
            Assert.AreEqual(layer.Height, deserialized.Height);
            Assert.AreEqual(layer.Pad, deserialized.Pad);
            Assert.AreEqual(layer.Stride, deserialized.Stride);
        }

        [TestMethod]
        public void RegressionLayerSerialization()
        {
            var layer = new RegressionLayer();
            layer.Init(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);


            var deserialized = LayerBase<double>.FromData(data) as RegressionLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
        }

        [TestMethod]
        public void ReluLayerSerialization()
        {
            var layer = new ReluLayer();
            layer.Init(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as ReluLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
        }

        [TestMethod]
        public void LeakyReluLayerSerialization()
        {
            var layer = new LeakyReluLayer();
            layer.Init(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as LeakyReluLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
        }

        [TestMethod]
        public void SigmoidLayerSerialization()
        {
            var layer = new SigmoidLayer();
            layer.Init(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as SigmoidLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
        }

        [TestMethod]
        public void SoftmaxLayerSerialization()
        {
            var layer = new SoftmaxLayer(10);
            layer.Init(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as SoftmaxLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);

            Assert.AreEqual(layer.ClassCount, deserialized.ClassCount);
        }

        [TestMethod]
        public void TanhLayerSerialization()
        {
            var layer = new TanhLayer();
            layer.Init(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as TanhLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);
        }
    }
}