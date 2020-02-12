using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Core.Fluent;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Serialization;
using NUnit.Framework;

namespace ConvNetSharp.Core.Tests
{
    /// <summary>
    ///     TODO: make it generic
    /// </summary>
    [TestFixture]
    public class SerializationTests
    {
        [Test]
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

        [Test]
        public void DropoutSerialization()
        {
            var layer = new DropoutLayer(0.1);
            layer.Init(28, 24, 1);
            var data = layer.GetData();

            Assert.AreEqual(28, data["InputWidth"]);
            Assert.AreEqual(24, data["InputHeight"]);
            Assert.AreEqual(1, data["InputDepth"]);

            var deserialized = LayerBase<double>.FromData(data) as DropoutLayer;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(28, deserialized.InputWidth);
            Assert.AreEqual(24, deserialized.InputHeight);
            Assert.AreEqual(1, deserialized.InputDepth);
            Assert.AreEqual(layer.OutputWidth, deserialized.OutputWidth);
            Assert.AreEqual(layer.OutputHeight, deserialized.OutputHeight);
            Assert.AreEqual(layer.OutputDepth, deserialized.OutputDepth);

            Assert.AreEqual(layer.DropProbability, deserialized.DropProbability);
        }

        [Test]
        public void FluentNetSerialization()
        {
            // Fluent version
            var net = FluentNet<double>.Create(24, 24, 1)
                .Conv(5, 5, 8).Stride(1).Pad(2)
                .Relu()
                .Pool(2, 2).Stride(2)
                .Conv(5, 5, 16).Stride(1).Pad(2)
                .Relu()
                .Pool(3, 3).Stride(3)
                .FullyConn(10)
                .Softmax(10)
                .Build();

            var json = net.ToJson();
            var deserialized = SerializationExtensions.FromJson<double>(json);

            Assert.AreEqual(9, deserialized.Layers.Count);
            Assert.IsTrue(deserialized.Layers[0] is InputLayer<double>);
            Assert.IsTrue(deserialized.Layers[8] is SoftmaxLayer<double>);
        }

        [Test]
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

        [Test]
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

        [Test]
        public void LeakyReluLayerSerialization()
        {
            var layer = new LeakyReluLayer(0.01);
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
            Assert.AreEqual(0.01, layer.Alpha);
        }

        [Test]
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

            Assert.AreEqual(8, deserialized.Layers.Count);
        }

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
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