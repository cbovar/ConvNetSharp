using ConvNetSharp.Fluent;
using System;
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
                var serializer = new DataContractJsonSerializer(typeof(Net), new DataContractJsonSerializerSettings { SerializeReadOnlyTypes = true, KnownTypes = new Type[] { typeof(Volume), typeof(VolumeWrapper) } });
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
                var serializer = new DataContractJsonSerializer(typeof(Net), new DataContractJsonSerializerSettings { SerializeReadOnlyTypes = true, KnownTypes = new Type[] { typeof(Volume), typeof(VolumeWrapper) } });

                Net net = serializer.ReadObject(ms) as Net;
                return net;
            }
        }

        public static INet LoadBinary(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            return formatter.Deserialize(stream) as INet;
        }

        public static void SaveBinary(this INet net, Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, net);
        }
    }
}
