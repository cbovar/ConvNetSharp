using System;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using Moq;
using NUnit.Framework;

namespace ConvNetSharp.Flow.Tests
{
    [TestFixture]
    public class OpsTests
    {
        public OpsTests()
        {
            BuilderInstance<double>.Volume = new VolumeBuilder();
        }

        private class MyOp : Op<float>
        {
            public MyOp(ConvNetSharp<float> graph) : base(graph)
            {
            }

            public override string Representation => "MyOp";

            public override void Differentiate()
            {
                throw new NotImplementedException();
            }

            public override Volume<float> Evaluate(Session<float> session)
            {
                throw new NotImplementedException();
            }
        }

        [Test]
        public void AddOpBackward()
        {
            var cns = new ConvNetSharp<double>();
            var volA = cns.Const(BuilderInstance<double>.Volume.SameAs(new Shape(1, 1, 3, 5)), "A");
            var volB = cns.Const(BuilderInstance<double>.Volume.From(new[] { 1.0, 2.0, 3.0 }, new Shape(1, 1, 3, 1)), "bias");
            var op = volA + volB;

            using var session = new Session<double>();

            var eval = op.Evaluate(session);
            Assert.IsNotNull(eval);

            op.Derivate = cns.Const(BuilderInstance<double>.Volume.From(new double[15].Populate(1.0), new Shape(1, 1, 3, 5)), "error");

            op.Differentiate();

            var volADiff = volA.Derivate.Evaluate(session);
            Assert.AreEqual(volA.Result.Shape, volADiff.Shape);
            var volBDiff = volB.Derivate.Evaluate(session);
            Assert.AreEqual(volB.Result.Shape, volBDiff.Shape);
        }

        [Test]
        public void AddOpForward()
        {
            var cns = new ConvNetSharp<double>();

            var nodeMockA = new Mock<Op<double>>(cns);
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volA);

            var nodeMockb = new Mock<Op<double>>(cns);
            var volB = new VolumeMock(2.0, new Shape(1));
            nodeMockb.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volB);

            var op = nodeMockA.Object + nodeMockb.Object;

            using var session = new Session<double>();

            var eval = op.Evaluate(session);
            Assert.IsNotNull(eval);

            var s = session;
            nodeMockA.Verify(o => o.Evaluate(s));
            nodeMockb.Verify(o => o.Evaluate(s));
            Assert.AreEqual(1, volA.DoAddCount);
        }

        [Test]
        public void AddParent()
        {
            var cns = new ConvNetSharp<float>();
            var op1 = new MyOp(cns);
            var op2 = new MyOp(cns);

            op1.AddParent(op2);

            Assert.IsTrue(op1.Parents.Contains(op2));
            Assert.IsTrue(op2.Children.Contains(op1));
        }

        [Test]
        public void MultOpForward()
        {
            var cns = new ConvNetSharp<double>();
            var nodeMockA = new Mock<Op<double>>(cns);
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volA);

            var nodeMockb = new Mock<Op<double>>(cns);
            var volB = new VolumeMock(2.0, new Shape(1));
            nodeMockb.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volB);

            var op = nodeMockA.Object * nodeMockb.Object;

            using var session = new Session<double>();

            var eval = op.Evaluate(session);

            Assert.IsNotNull(eval);

            var s = session;
            nodeMockA.Verify(o => o.Evaluate(s));
            nodeMockb.Verify(o => o.Evaluate(s));
            Assert.AreEqual(1, volA.DoMultiplyCount);
        }

        [Test]
        public void NegateOpForward()
        {
            var cns = new ConvNetSharp<double>();
            var nodeMockA = new Mock<Op<double>>(cns);
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volA);

            var op = -nodeMockA.Object;

            using var session = new Session<double>();

            var eval = op.Evaluate(session);

            Assert.IsNotNull(eval);

            var s = session;
            nodeMockA.Verify(o => o.Evaluate(s));
            Assert.AreEqual(1, volA.DoNegateCount);
        }

        [Test]
        public void RemoveParent()
        {
            var cns = new ConvNetSharp<float>();
            var op1 = new MyOp(cns);
            var op2 = new MyOp(cns);

            op1.AddParent(op2);
            op1.RemoveParent(op2);

            Assert.IsFalse(op1.Parents.Contains(op2));
            Assert.IsFalse(op2.Children.Contains(op1));
        }

        [Test]
        public void Scope()
        {
            var cns = new ConvNetSharp<float>();

            var v0 = cns.Variable("0");
            Assert.AreEqual("0", v0.Name);

            using (cns.Scope("layer1"))
            {
                var v1 = cns.Variable("A");
                Assert.AreEqual("layer1/A", v1.Name);

                using (cns.Scope("linear"))
                {
                    var v2 = cns.Variable("B");
                    Assert.AreEqual("layer1/linear/B", v2.Name);
                }

                var v3 = cns.Variable("C");
                Assert.AreEqual("layer1/C", v3.Name);
            }
        }
    }
}