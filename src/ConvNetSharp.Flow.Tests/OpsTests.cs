using System;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public class OpsTests
    {
        public OpsTests()
        {
            BuilderInstance<double>.Volume = new Volume.GPU.Double.VolumeBuilder();
        }

        class MyOp : Op<float>
        {
            public override void Differentiate()
            {
                throw new System.NotImplementedException();
            }

            public override Volume<float> Evaluate(Session<float> session)
            {
                throw new System.NotImplementedException();
            }

            public override string Representation { get; }
        }

        [TestMethod]
        public void AddParent()
        {
            var op1 = new MyOp();
            var op2 = new MyOp();

            op1.AddParent(op2);

            Assert.IsTrue(op1.Parents.Contains(op2));
            Assert.IsTrue(op2.Children.Contains(op1));
        }

        [TestMethod]
        public void RemoveParent()
        {
            var op1 = new MyOp();
            var op2 = new MyOp();

            op1.AddParent(op2);

            op1.RemoveParent(op2);

            Assert.IsFalse(op1.Parents.Contains(op2));
            Assert.IsFalse(op2.Children.Contains(op1));
        }

        [TestMethod]
        public void AddOpForward()
        {
            var nodeMockA = new Mock<Op<double>>();
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volA);

            var nodeMockb = new Mock<Op<double>>();
            var volB = new VolumeMock(2.0, new Shape(1));
            nodeMockb.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volB);

            var op = new Add<double>(nodeMockA.Object, nodeMockb.Object);

            using (var session = new Session<double>())
            {
                var eval = op.Evaluate(session);
                Assert.IsNotNull(eval);

                var s = session;
                nodeMockA.Verify(o => o.Evaluate(s));
                nodeMockb.Verify(o => o.Evaluate(s));
                Assert.AreEqual(1, volA.DoAddCount);
            }
        }

        [TestMethod]
        public void AddOpBackward()
        {
            var volA = new Const<double>(BuilderInstance<double>.Volume.SameAs(new Shape(1, 1, 3, 5)), "A");
            var volB = new Const<double>(BuilderInstance<double>.Volume.From(new[] { 1.0, 2.0, 3.0 }, new Shape(1, 1, 3, 1)), "bias");
            var op = new Add<double>(volA, volB);

            using (var session = new Session<double>())
            {
                var eval = op.Evaluate(session);
                Assert.IsNotNull(eval);

                op.Derivate = new Const<double>(BuilderInstance<double>.Volume.From(new double[15].Populate(1.0), new Shape(1, 1, 3, 5)), "error");

                op.Differentiate();

                var volADiff = volA.Derivate.Evaluate(session);
                Assert.AreEqual(volA.Result.Shape, volADiff.Shape);
                var volBDiff = volB.Derivate.Evaluate(session);
                Assert.AreEqual(volB.Result.Shape, volBDiff.Shape);
            }
        }

        [TestMethod]
        public void MultOpForward()
        {
            var nodeMockA = new Mock<Op<double>>();
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volA);

            var nodeMockb = new Mock<Op<double>>();
            var volB = new VolumeMock(2.0, new Shape(1));
            nodeMockb.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volB);

            var op = new Mult<double>(nodeMockA.Object, nodeMockb.Object);

            using (var session = new Session<double>())
            {
                var eval = op.Evaluate(session);

                Assert.IsNotNull(eval);

                var s = session;
                nodeMockA.Verify(o => o.Evaluate(s));
                nodeMockb.Verify(o => o.Evaluate(s));
                Assert.AreEqual(1, volA.DoMultiplyCount);
            }
        }

        [TestMethod]
        public void NegateOpForward()
        {
            var nodeMockA = new Mock<Op<double>>();
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volA);

            var op = new Negate<double>(nodeMockA.Object);

            using (var session = new Session<double>())
            {
                var eval = op.Evaluate(session);

                Assert.IsNotNull(eval);

                var s = session;
                nodeMockA.Verify(o => o.Evaluate(s));
                Assert.AreEqual(1, volA.DoNegateCount);
            }
        }

        [TestMethod]
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

        [TestMethod]
        public void SumOp()
        {
            var x = new Const<float>(BuilderInstance<float>.Volume.From(new[] { 1.0f, 2.0f, 3.0f }, new Shape(3)), "x");
            var op = new Sum<float>(x, new Shape(1));

            using (var session = new Session<float>())
            {
                var result = op.Evaluate(session);
                Assert.AreEqual(6.0f, result.Get(0));
            }
        }

        [TestMethod]
        public void SumOpDerivative()
        {
            var x = new Const<float>(BuilderInstance<float>.Volume.From(new float[] { 1.0f, 2.0f, 3.0f }, new Shape(3)), "x");
            var op = new Sum<float>(x, new Shape(1));

            using (var session = new Session<float>())
            {
                session.Differentiate(op);

                op.Derivate = new Const<float>(50.0f, "50");

                var result = x.Derivate.Evaluate(session);
            }
        }
    }
}