using System.IO;

namespace ConvNetSharp.Serialization
{
    public interface INetSerializer
    {
        void Save(Net net, Stream stream);

        Net Load(Stream stream);
    }
}
