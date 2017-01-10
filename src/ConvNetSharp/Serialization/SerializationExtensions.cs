using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Runtime.Serialization.Json;
using System.Text;

namespace ConvNetSharp.Serialization
{
    public static class SerializationExtensions
    {
        public static string ToJSON(this Net net)
        {
            //Serializes net to JSON
            using (var ms = new MemoryStream())
            {
                var serializer = new DataContractJsonSerializer(typeof(Net), new DataContractJsonSerializerSettings { SerializeReadOnlyTypes = true });
                serializer.WriteObject(ms, net);
                ms.Position = 0;

                StreamReader sr = new StreamReader(ms);
                return sr.ReadToEnd();
            }
        }

        public static Net FromJSON(string json)
        {
            using (var ms = new MemoryStream(Encoding.UTF8.GetBytes(json)))
            {
                //Deserializes JSON to net
                ms.Position = 0;
                var serializer = new DataContractJsonSerializer(typeof(Net));

                Net net = serializer.ReadObject(ms) as Net;
                return net;
            }
        }

        public static Net LoadBinary(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            return formatter.Deserialize(stream) as Net;
        }

        public static void SaveBinary(this Net net, Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, net);
        }
    }
}
