using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Flow.Serialization;
using ConvNetSharp.Volume;
using NUnit.Framework;

namespace ConvNetSharp.Flow.Tests
{
    [TestFixture]
    public class SerializationTests
    {
        [Test]
        public void Activation()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = new Activation<double>(cns, x, ActivationType.Relu);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Activation<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(ActivationType.Relu, deserialized.Type);
        }

        [Test]
        public void Add()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var b = cns.Const(2.0, "two");
            var op = a + b;

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Add<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("two", (deserialized.Parents[1] as Const<double>).Name);
        }

        [Test]
        public void Const()
        {
            var cns = new ConvNetSharp<double>();
            var op = cns.Const(1.0, "one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Const<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }

        [Test]
        public void Convolution()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = new Convolution<double>(cns, x, 5, 5, 16, 2, 1);

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

        [Test]
        public void Dense()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = new Dense<double>(cns, x, 16);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Dense<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count); // input and filters
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(1, deserialized.Width);
            Assert.AreEqual(1, deserialized.Height);
            Assert.AreEqual(16, deserialized.FilterCount);
            Assert.AreEqual(1, deserialized.Stride);
            Assert.AreEqual(0, deserialized.Pad);
        }

        [Test]
        public void Div()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var b = cns.Const(2.0, "two");
            var op = a / b;

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Div<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("two", (deserialized.Parents[1] as Const<double>).Name);
        }

        [Test]
        public void Dropout()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var dropoutProbability = 0.5;
            var op = cns.Dropout(x, dropoutProbability);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Dropout<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(dropoutProbability, ((Const<double>)deserialized.DropoutProbability).Value);
        }

        [Test]
        public void Exp()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var op = cns.Exp(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Exp<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void GraphMl()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var b = cns.Const(2.0, "two");
            var add = a + b;
            var activation = cns.Relu(add);

            activation.Save("test");

            var result = SerializationExtensions.Load<double>("test", false);
        }

        [Test]
        public void LeakyRelu()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var op = cns.LeakyRelu(a, 0.01);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as LeakyRelu<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void Log()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var op = cns.Log(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Log<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void MatMult()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var b = cns.Const(2.0, "two");
            var op = cns.MatMult(a, b);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as MatMult<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("two", (deserialized.Parents[1] as Const<double>).Name);
        }

        [Test]
        public void Max()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var op = cns.Max(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Max<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void Mult()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var b = cns.Const(2.0, "two");
            var op = a * b;

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Mult<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("two", (deserialized.Parents[1] as Const<double>).Name);
        }

        [Test]
        public void Negate()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = cns.Negate(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Negate<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void PlaceHolder()
        {
            var cns = new ConvNetSharp<double>();
            var op = cns.PlaceHolder("one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as PlaceHolder<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }

        [Test]
        public void Pool()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = cns.Pool(x, 3, 4, 1, 2, 1, 2);

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

        [Test]
        public void Reshape1()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = cns.Reshape(x, new Shape(1, 2, 3, 4)) as Reshape<double>;

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Reshape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(op.OutputShape, deserialized.OutputShape);
        }

        [Test]
        public void Reshape2()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var shape = cns.Const(new[] { 1.0, 2.0, 3.0, 4.0 }, "shape");
            var op = cns.Reshape(x, shape);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Reshape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("shape", (deserialized.Parents[1] as Const<double>).Name);
        }

        [Test]
        public void Shape()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = cns.Shape(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Shape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void ShapeIndex()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = new Shape<double>(cns, x, 3);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Shape<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual(3, deserialized.Index);
        }

        [Test]
        public void Softmax()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var op = cns.Softmax(x);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Softmax<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("x", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void SoftmaxCrossEntropy()
        {
            var cns = new ConvNetSharp<double>();
            var softmax = cns.Const(1.0, "softmax");
            var y = cns.Const(1.0, "y");
            var op = new SoftmaxCrossEntropy<double>(cns, softmax, y);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as SoftmaxCrossEntropy<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(2, deserialized.Parents.Count);
            Assert.AreEqual("softmax", (deserialized.Parents[0] as Const<double>).Name);
            Assert.AreEqual("y", (deserialized.Parents[1] as Const<double>).Name);
        }

        [Test]
        public void Sqrt()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(9.0, "input");
            var op = cns.Sqrt(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Sqrt<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("input", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void Tile()
        {
            var cns = new ConvNetSharp<double>();
            var x = cns.Const(1.0, "x");
            var a = cns.Const(1.0, "a");
            var op = cns.Tile(x, cns.Shape(a));

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Tile<double>;

            Assert.IsNotNull(deserialized);
        }

        [Test]
        public void Transpose()
        {
            var cns = new ConvNetSharp<double>();
            var a = cns.Const(1.0, "one");
            var op = cns.Transpose(a);

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Transpose<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(1, deserialized.Parents.Count);
            Assert.AreEqual("one", (deserialized.Parents[0] as Const<double>).Name);
        }

        [Test]
        public void Variable()
        {
            var cns = new ConvNetSharp<double>();
            var op = cns.Variable(1.0, "one");

            var xml = op.ToXml();
            var deserialized = SerializationExtensions.FromXml<double>(xml) as Variable<double>;

            Assert.IsNotNull(deserialized);
            Assert.AreEqual(op.Name, deserialized.Name);
        }
    }
}