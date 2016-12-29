using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace ConvNetSharp.Serialization
{
    public class BinaryNetSerializer : INetSerializer
    {
        //public void Save(string filename)
        //{
        //    using (var fs = new FileStream(filename, FileMode.Create))
        //    {
        //        IFormatter formatter = new BinaryFormatter();
        //        formatter.Serialize(fs, this);
        //    }
        //}

        //public static Net Load(string filename)
        //{
        //    Net result = null;
        //    if (File.Exists(filename))
        //    {
        //        using (var fs = new FileStream(filename, FileMode.Open))
        //        {
        //            IFormatter formatter = new BinaryFormatter();
        //            result = formatter.Deserialize(fs) as Net;
        //        }
        //    }

        //    return result;
        //}

        public Net Load(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            return formatter.Deserialize(stream) as Net;
        }

        public void Save(Net net, Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, net);
        }
    }
}
