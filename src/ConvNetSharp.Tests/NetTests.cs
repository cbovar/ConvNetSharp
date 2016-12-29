using ConvNetSharp.Serialization;
using Moq;
using NUnit.Framework;
using System.IO;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class NetTests
    {
        [Test]
        public void SerializerIsUsedTest()
        {
            var serializerMock = new Mock<INetSerializer>();

            var net = new Net();

            using (var ms = new MemoryStream())
            {
                // Serialization
                net.Save(serializerMock.Object, ms);
                serializerMock.Verify(o => o.Save(net, ms), Times.Once);

                // Deserialization
                ms.Position = 0;
                Net.Load(serializerMock.Object, ms);
                serializerMock.Verify(o => o.Load(ms), Times.Once);
            }
        }
    }
}
