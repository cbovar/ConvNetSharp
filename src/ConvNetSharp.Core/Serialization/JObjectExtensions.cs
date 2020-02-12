using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace ConvNetSharp.Core.Serialization
{
    public static class JObjectExtensions
    {
        /// <summary>
        ///     JObject to nested dictionaries
        /// </summary>
        /// <param name="jobject"></param>
        /// <returns></returns>
        public static Dictionary<string, object> ToDictionary(this JObject jobject)
        {
            var dico = new Dictionary<string, object>();

            foreach (var o in jobject)
            {
                var oValue = o.Value;

                if (o.Value is JArray jArray)
                {
                    var first = jArray[0]; // use first element to guess if we are dealing with a list of dico or an array
                    var isValueArray = first is JValue;

                    if (isValueArray)
                    {
                        var array = jArray.Values().Select(x => ((JValue) x).Value).ToArray();
                        dico[o.Key] = array;
                    }
                    else
                    {
                        var list = new List<IDictionary<string, object>>();
                        foreach (var token in jArray)
                        {
                            var elt = ((JObject) token).ToDictionary();
                            list.Add(elt);
                        }

                        dico[o.Key] = list;
                    }
                }
                else
                {
                    dico[o.Key] = ((JValue) oValue).Value;
                }
            }

            return dico;
        }
    }
}