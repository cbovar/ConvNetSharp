﻿using System.IO;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Flow.Serialization;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public class SerializationTests
    {
        [TestMethod]
        public void Activation()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var op = new Activation<double>(graph, x, ActivationType.Relu);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Activation<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(ActivationType.Relu, deserialized.Type);
        }

        [TestMethod]
        public void Add()
        {
            var graph = new ConvNetSharp<double>();
            var a = graph.Const(1.0, "one");
            var b = graph.Const(2.0, "two");
            var op = a + b;

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Add<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("two", (deserialized.Parents[1] as Const<double>).Name);
        }

        [TestMethod]
        public void Const()
        {
            var graph = new ConvNetSharp<double>();
            var op = graph.Const(1.0, "one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Const<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }

        [TestMethod]
        public void Convolution()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var op = new Convolution<double>(graph, x, 5, 5, 16, 2, 1);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Convolution<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count); // input and filters
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(5, deserialized.Width);
            Assert.AreEqual(5, deserialized.Height);
            Assert.AreEqual(16, deserialized.FilterCount);
            Assert.AreEqual(2, deserialized.Stride);
            Assert.AreEqual(1, deserialized.Pad);
        }

        [TestMethod]
        public void Div()
        {
            var graph = new ConvNetSharp<double>();
            var a = graph.Const(1.0, "one");
            var b = graph.Const(2.0, "two");
            var op = a / b;

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Div<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("two", (deserialized.Parents[1] as Const<double>).Name);
        }

        [TestMethod]
        public void Exp()
        {
            var graph = new ConvNetSharp<double>();
            var a = graph.Const(1.0, "one");
            var op = graph.Exp(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Exp<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void Sqrt()
        {
            var graph = new ConvNetSharp<double>();
            var a = graph.Const(9.0, "input");
            var op = graph.Sqrt(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Sqrt<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("input", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void GraphMl()
        {
            var graph = new ConvNetSharp<double>();
            var a = graph.Const(1.0, "one");
            var b = graph.Const(2.0, "two");
            var add = a + b;
            var activation = graph.Relu(add);

            activation.Save("test");

            var result = SerializationExtensions.Load<double>("test", false);
        }

        [TestMethod]
        public void Log()
        {
            var graph = new ConvNetSharp<double>();
            var a = graph.Const(1.0, "one");
            var op = graph.Log(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Log<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void Max()
        {
            var graph = new ConvNetSharp<double>();
            var a = graph.Const(1.0, "one");
            var op = graph.Max(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Max<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void Mult()
        {
            var graph = new ConvNetSharp<double>();
            var a = graph.Const(1.0, "one");
            var b = graph.Const(2.0, "two");
            var op = a * b;

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Mult<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("two", (deserialized.Parents[1] as Const<double>).Name);
        }

        [TestMethod]
        public void Negate()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var op = graph.Negate(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Negate<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void PlaceHolder()
        {
            var graph = new ConvNetSharp<double>();
            var op = graph.PlaceHolder("one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as PlaceHolder<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }

        [TestMethod]
        public void Tile()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var a = graph.Const(1.0, "a");
            var op = graph.Tile(x, graph.Shape(a));

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Tile<double>;

            Assert.IsNotNull(deserialized);
        }

        [TestMethod]
        public void Dropout()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var dropoutProbability = 0.5;
            var op = graph.Dropout(x, dropoutProbability);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Dropout<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(dropoutProbability, deserialized.DropoutProbability);
        }

        [TestMethod]
        public void Pool()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var op = graph.Pool(x, 3, 4, 1, 2, 1, 2);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Pool<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(3, deserialized.Width);
            Assert.AreEqual(4, deserialized.Height);
            Assert.AreEqual(1, deserialized.HorizontalPad);
            Assert.AreEqual(2, deserialized.VerticalPad);
            Assert.AreEqual(1, deserialized.HorizontalStride);
            Assert.AreEqual(2, deserialized.VerticalStride);
        }

        [TestMethod]
        public void Reshape1()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var op = graph.Reshape(x, new Shape(1, 2, 3, 4)) as Reshape<double>;

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Reshape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(op.OutputShape, deserialized.OutputShape);
        }

        [TestMethod]
        public void Reshape2()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var shape = graph.Const(new[]{1.0,2.0,3.0,4.0}, "shape");
            var op = graph.Reshape(x, shape);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Reshape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("shape", (deserialized.Parents[1] as Const<double>).Name);
        }

        [TestMethod]
        public void Shape()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var op = graph.Shape(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Shape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void ShapeIndex()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var op = new Shape<double>(graph, x, 3);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Shape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(3, deserialized.Index);
        }

        [TestMethod]
        public void Softmax()
        {
            var graph = new ConvNetSharp<double>();
            var x = graph.Const(1.0, "x");
            var op = graph.Softmax(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Softmax<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void SoftmaxCrossEntropy()
        {
            var graph = new ConvNetSharp<double>();
            var softmax = graph.Const(1.0, "softmax");
            var y = graph.Const(1.0, "y");
            var op = new SoftmaxCrossEntropy<double>(graph, softmax, y);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as SoftmaxCrossEntropy<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("softmax", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("y", (deserialized.Parents[1] as Const<double>).Name);
        }

        [TestMethod]
        public void Variable()
        {
            var graph = new ConvNetSharp<double>();
            var op = graph.Variable(1.0, "one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Variable<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }
    }
}