using ConvNetSharp.Core.Ops;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace ConvNetSharp.Core.Tests
{
    [TestClass]
    public class OpsTests
    {
        [TestMethod]
        public void AddOpForward()
        {
            var nodeMockA = new Mock<Op<double>>();
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Forward(null)).Returns(volA);

            var nodeMockb = new Mock<Op<double>>();
            var volB = new VolumeMock(2.0, new Shape(1));
            nodeMockb.Setup(o => o.Forward(null)).Returns(volB);

            var op = new AddOp<double> {Parents = {nodeMockA.Object, nodeMockb.Object}};

            var eval = op.Forward(null);

            Assert.IsNotNull(eval);
            nodeMockA.Verify(o => o.Forward(null));
            nodeMockb.Verify(o => o.Forward(null));
            Assert.AreEqual(1, volA.DoAddCount);
        }

        [TestMethod]
        public void MultOpForward()
        {
            var nodeMockA = new Mock<Op<double>>();
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Forward(null)).Returns(volA);

            var nodeMockb = new Mock<Op<double>>();
            var volB = new VolumeMock(2.0, new Shape(1));
            nodeMockb.Setup(o => o.Forward(null)).Returns(volB);

            var op = new MultOp<double> {Parents = {nodeMockA.Object, nodeMockb.Object}};

            var eval = op.Forward(null);

            Assert.IsNotNull(eval);
            nodeMockA.Verify(o => o.Forward(null));
            nodeMockb.Verify(o => o.Forward(null));
            Assert.AreEqual(1, volA.DoMultiplyCount);
        }

        [TestMethod]
        public void NegateOpForward()
        {
            var nodeMockA = new Mock<Op<double>>();
            var volA = new VolumeMock(1.0, new Shape(1));
            nodeMockA.Setup(o => o.Forward(null)).Returns(volA);

            var op = new NegateOp<double> {Parents = {nodeMockA.Object}};

            var eval = op.Forward(null);

            Assert.IsNotNull(eval);
            nodeMockA.Verify(o => o.Forward(null));
            Assert.AreEqual(1, volA.DoNegateCount);
        }
    }
}