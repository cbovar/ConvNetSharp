using System;
using System.Linq;
using ConvNetSharp.Core.Fluent;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace ConvNetSharp.Core.Serialization
{
    public static class SerializationExtensions
    {
        public static Net<T> FromJson<T>(string json) where T : struct, IEquatable<T>, IFormattable
        {
            var data = JsonConvert.DeserializeObject<JObject>(json);
            var dico = data.ToDictionary();
            var net = Net<T>.FromData(dico);
            return net;
        }

        public static T[] ToArrayOfT<T>(this object obj)
        {
            if (obj is T[] arrayofT)
            {
                return arrayofT;
            }

            if (obj is JArray jarray)
            {
                return jarray.ToObject<T[]>();
            }

            return ((object[])obj).Select(o => (T)Convert.ChangeType(o, typeof(T), null)).ToArray();
        }

        public static string ToJson<T>(this Net<T> net) where T : struct, IEquatable<T>, IFormattable
        {
            var data = net.GetData();
            var json = JsonConvert.SerializeObject(data);
            return json;
        }

        public static string ToJson<T>(this FluentNet<T> net) where T : struct, IEquatable<T>, IFormattable
        {
            var data = net.GetData();
            var json = JsonConvert.SerializeObject(data);
            return json;
        }
    }
}