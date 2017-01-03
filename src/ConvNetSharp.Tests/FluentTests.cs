using NUnit.Framework;
using ConvNetSharp.Fluent;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class FluentTests
    {
        [Test]
        public void CreateTest()
        {
            var net = FluentNet.Create(10, 10, 2)
                .Relu()
                .FullyConn(10)
                .SoftMax(10)
                .Build();

            //net.Forward(new Volume(10, 10, 2));
        }

        [Test]
        public void MergeTest()
        {
            var branch1 = FluentNet.Create(10, 10, 2)
                .Relu()
                .FullyConn(10);
            var branch2 = FluentNet.Create(10, 10, 2)
                .Relu()
                .FullyConn(20);

            var net = FluentNet.Merge(branch1, branch2)
                .FullyConn(5)
                .SoftMax(5)
                .Build();

            net.InputLayers[0].Forward(new Volume(10, 10, 2));
            net.InputLayers[1].Forward(new Volume(10, 10, 2));

            net.Forward();
        }
    }
}
