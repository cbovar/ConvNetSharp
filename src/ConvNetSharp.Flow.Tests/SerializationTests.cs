using System.IO;
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
            var x = new Const<double>(1.0, "x");
            var op = new Activation<double>(x, ActivationType.Relu);

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
            var a = new Const<double>(1.0, "one");
            var b = new Const<double>(2.0, "two");
            var op = new Add<double>(a, b);

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
            var op = new Const<double>(1.0, "one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Const<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }

        [TestMethod]
        public void Convolution()
        {
            var x = new Const<double>(1.0, "x");
            var op = new Convolution<double>(x, 5, 5, 16, 2, 1);

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
            var a = new Const<double>(1.0, "one");
            var b = new Const<double>(2.0, "two");
            var op = new Div<double>(a, b);

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
            var a = new Const<double>(1.0, "one");
            var op = new Exp<double>(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Exp<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void GraphMl()
        {
            var a = new Const<double>(1.0, "one");
            var b = new Const<double>(2.0, "two");
            var add = new Add<double>(a, b);
            var activation = new Activation<double>(add, ActivationType.Relu);

            activation.Save("test");

            var result = SerializationExtensions.Load<double>("test", false);
        }

        [TestMethod]
        public void Log()
        {
            var a = new Const<double>(1.0, "one");
            var op = new Log<double>(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Log<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void Max()
        {
            var a = new Const<double>(1.0, "one");
            var op = new Max<double>(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Max<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void Mult()
        {
            var a = new Const<double>(1.0, "one");
            var b = new Const<double>(2.0, "two");
            var op = new Mult<double>(a, b);

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
            var x = new Const<double>(1.0, "x");
            var op = new Negate<double>(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Negate<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void PlaceHolder()
        {
            var op = new PlaceHolder<double>("one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as PlaceHolder<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }

        [TestMethod]
        public void Pool()
        {
            var x = new Const<double>(1.0, "x");
            var op = new Pool<double>(x, 3, 4, 1, 2, 1, 2);

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
            var x = new Const<double>(1.0, "x");
            var op = new Reshape<double>(x, new Shape(1, 2, 3, 4));

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
            var x = new Const<double>(1.0, "x");
            var shape = new Const<double>(new[]{1.0,2.0,3.0,4.0}, "shape");
            var op = new Reshape<double>(x, shape);

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
            var x = new Const<double>(1.0, "x");
            var op = new Shape<double>(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Shape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void Softmax()
        {
            var x = new Const<double>(1.0, "x");
            var op = new Softmax<double>(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Softmax<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [TestMethod]
        public void SoftmaxCrossEntropy()
        {
            var softmax = new Const<double>(1.0, "softmax");
            var y = new Const<double>(1.0, "y");
            var op = new SoftmaxCrossEntropy<double>(softmax, y);

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
            var op = new Variable<double>(1.0, "one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Variable<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }
    }
}