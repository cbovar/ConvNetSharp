using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public class OpsTests
    {
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

            var op = new AddOp<double>(nodeMockA.Object, nodeMockb.Object);

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
        public void MultOpForward()
        {
            var nodeMockA = new Mock<Op<double>>();
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volA);

            var nodeMockb = new Mock<Op<double>>();
            var volB = new VolumeMock(2.0, new Shape(1));
            nodeMockb.Setup(o => o.Evaluate(It.IsAny<Session<double>>())).Returns(volB);

            var op = new MultOp<double>(nodeMockA.Object, nodeMockb.Object);

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

            var op = new NegateOp<double>(nodeMockA.Object);

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

            var v0 = cns.Variable(null, "0");
            Assert.AreEqual("0", v0.Name);

            using (cns.Scope("layer1"))
            {
                var v1 = cns.Variable(null, "A");
                Assert.AreEqual("layer1/A", v1.Name);

                using (cns.Scope("linear"))
                {
                    var v2 = cns.Variable(null, "B");
                    Assert.AreEqual("layer1/linear/B", v2.Name);
                }

                var v3 = cns.Variable(null, "C");
                Assert.AreEqual("layer1/C", v3.Name);
            }
        }
    }
}