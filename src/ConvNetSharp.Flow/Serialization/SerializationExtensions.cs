using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Xml;
using ConvNetSharp.Core.Serialization;
using ConvNetSharp.Flow.Ops;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Serialization
{
    public static class SerializationExtensions
    {
        public static Op<T> FromXml<T>(string xml) where T : struct, IEquatable<T>, IFormattable
        {
            return FromXml<T>(xml, false)[0];
        }

        /// <summary>
        ///     Deserialize graph from graphml file
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="xml"></param>
        /// <param name="includeCost">if true the returned list will contain two Ops: [root, cost]</param>
        /// <returns></returns>
        public static List<Op<T>> FromXml<T>(string xml, bool includeCost) where T : struct, IEquatable<T>, IFormattable
        {
            var root = "";
            var cost = "";
            var keys = new Dictionary<string, KeyDescription>();
            var nodes = new Dictionary<string, Node>();
            var edges = new List<Edge>();

            using (var sw = new StringReader(xml))
            {
                using (var reader = XmlReader.Create(sw))
                {
                    reader.MoveToContent();
                    while (reader.Read())
                    {
                        if (reader.NodeType == XmlNodeType.Element)
                        {
                            switch (reader.Name)
                            {
                                case "key":
                                    {
                                        var id = reader.GetAttribute("id");
                                        var name = reader.GetAttribute("attr.name");
                                        var firstDot = name.IndexOf('.');
                                        if (firstDot != -1)
                                        {
                                            name = name.Substring(name.IndexOf('.') + 1, name.Length - name.IndexOf('.') - 1);
                                        }

                                        keys[id] = new KeyDescription { Id = id, Name = name };
                                    }
                                    break;

                                case "node":
                                    {
                                        var id = reader.GetAttribute("id");
                                        var node = new Node { Id = id };
                                        nodes[id] = node;

                                        // Move to data section
                                        do
                                        {
                                            reader.Read();
                                        } while (reader.Name != "data");

                                        // Parse data
                                        do
                                        {
                                            var key = reader.GetAttribute("key");
                                            var keyDesc = keys[key];
                                            node.Data[keyDesc.Name] = reader.ReadElementContentAsString();

                                            reader.Read();
                                        } while (reader.Name == "data");
                                    }
                                    break;

                                case "edge":
                                    {
                                        var source = reader.GetAttribute("source");
                                        var target = reader.GetAttribute("target");
                                        edges.Add(new Edge { Source = source, Target = target });
                                    }
                                    break;

                                case "data":
                                    {
                                        var key = reader.GetAttribute("key");
                                        var keyDesc = keys[key];
                                        if (keyDesc.Name == "root")
                                        {
                                            root = reader.ReadElementContentAsString();
                                        }
                                        else if (keyDesc.Name == "cost")
                                        {
                                            cost = reader.ReadElementContentAsString();
                                        }
                                    }
                                    break;
                            }
                        }
                    }
                }
            }

            // Create Ops
            var ops = new Dictionary<string, Op<T>>();
            foreach (var node in nodes)
            {
                var type = Type.GetType((string)node.Value.Data["type"]);
                var op = (Op<T>)Activator.CreateInstance(type, node.Value.Data);
                ops[node.Key] = op;
            }

            // Link Ops
            foreach (var edge in edges)
            {
                var source = ops[edge.Source];
                var target = ops[edge.Target];

                target.AddParent(source);
            }

            var result = new List<Op<T>> { ops[root] };
            if (includeCost)
            {
                if (cost != null)
                {
                    result.Add(ops[cost]);
                }
            }
            return result;
        }

        private static Volume<T> BuildVolume<T>(Dictionary<string, object> dico) where T : struct, IEquatable<T>, IFormattable
        {
            var dim0 = Convert.ToInt32(dico["dim0"]);
            var dim1 = Convert.ToInt32(dico["dim1"]);
            var dim2 = Convert.ToInt32(dico["dim2"]);
            var dim3 = Convert.ToInt32(dico["dim3"]);
            var shape = new Shape(dim0, dim1, dim2, dim3);
            var data = dico["vol"].ToArrayOfT<T>();

            return BuilderInstance<T>.Volume.From(data, shape);
        }

        public static List<Op<T>> Load<T>(string name, bool includeCost) where T : struct, IEquatable<T>, IFormattable
        {
            List<Op<T>> result;
            using (var sr = new StreamReader(File.Open($"{name}.graphml", FileMode.Open)))
            {
                var graphml = sr.ReadToEnd();
                result = FromXml<T>(graphml, includeCost);
            }

            using (var sr = new StreamReader(File.Open($"{name}.json", FileMode.Open)))
            {
                var json = sr.ReadToEnd();
                var data = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, object>>>(json);

                // Find all PlaceHolders and update their current value
                var visitor = new OpVisitor<T>(op =>
                {
                    var variable = op as IPersistable<T>;
                    if (variable != null)
                    {
                        Dictionary<string, object> d;
                        if (data.TryGetValue(variable.Name, out d))
                        {
                            variable.Result = BuildVolume<T>(d);
                        }
                    }
                });

                result[0].Accept(visitor);
                if (result.Count == 2)
                {
                    result[1]?.Accept(visitor);
                }
            }

            return result;
        }

        public static Dictionary<string, object> GetVolumes<T>(this Op<T> op) where T : struct, IEquatable<T>, IFormattable
        {
            var result = new Dictionary<string, object>();

            // Retrieve all ops and assign an Id
            var visitor = new OpVisitor<T>(o =>
            {
                var persistable = o as IPersistable<T>;
                var volume = persistable?.Result;
                if (volume != null)
                {
                    var dico = new Dictionary<string, object>
                    {
                        ["dim0"] = volume.Shape.GetDimension(0),
                        ["dim1"] = volume.Shape.GetDimension(1),
                        ["dim2"] = volume.Shape.GetDimension(2),
                        ["dim3"] = volume.Shape.GetDimension(3),
                        ["vol"] = volume.ToArray(),
                    };

                    result.Add(persistable.Name, dico); // we use Add here to it throws when the name is already used
                }
            });
            op.Accept(visitor);

            return result;
        }

        public static void Save<T>(this Op<T> op, string name, Op<T> costOp = null) where T : struct, IEquatable<T>, IFormattable
        {
            using (var sw = new StreamWriter(File.Create($"{name}.graphml")))
            {
                var graphml = op.ToXml(costOp);
                sw.Write(graphml);
            }

            using (var sw = new StreamWriter(File.Create($"{name}.json")))
            {
                var data = op.GetVolumes();
                var costData = costOp == null ? new Dictionary<string, object>() : costOp.GetVolumes();

                var result = data.Concat(costData.Where(x => !data.Keys.Contains(x.Key))).ToDictionary(pair => pair.Key, pair => pair.Value);
                //new[] { data, costData }.SelectMany(dict => dict)
                //.ToDictionary(pair => pair.Key, pair => pair.Value);

                var json = JsonConvert.SerializeObject(result);
                sw.Write(json);
            }
        }

        /// <summary>
        ///     Serialize graph to graphml
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="op">Root op</param>
        /// <param name="costOp">Optional cost Op</param>
        /// <returns></returns>
        public static string ToXml<T>(this Op<T> op, Op<T> costOp = null) where T : struct, IEquatable<T>, IFormattable
        {
            var id = 0;
            var set = new Dictionary<Op<T>, string>();
            var keys = new Dictionary<string, KeyDescription>
            {
                {"d0", new KeyDescription {Id = "d0", Name = "type"}},
                {"d1", new KeyDescription {Id = "d1", Name = "root", @for = "graph"}},
                {"d2", new KeyDescription {Id = "d2", Name = "cost", @for = "graph"}}
            };

            // Retrieve all ops and assign an Id
            var visitor = new OpVisitor<T>(o =>
            {
                if (!set.ContainsKey(o))
                {
                    set.Add(o, "n" + id++);
                }
            });
            op.Accept(visitor);
            costOp?.Accept(visitor);

            using (var sw = new StringWriter())
            {
                using (var writer = XmlWriter.Create(sw, new XmlWriterSettings { NewLineOnAttributes = true, Indent = true }))
                {
                    var ns = "http://graphml.graphdrawing.org/xmlns";

                    writer.WriteStartDocument();
                    writer.WriteStartElement("graphml", ns);
                    writer.WriteAttributeString("xmlns", ns);

                    // Get all keys
                    var keyId = 3;
                    foreach (var pair in set)
                    {
                        var data = pair.Key.GetData();
                        if (data.Any())
                        {
                            foreach (var o in data)
                            {
                                var name = pair.Key.GetType().Name + "." + o.Key;
                                if (!keys.ContainsKey(name))
                                {
                                    keys[name] = new KeyDescription { Id = "d" + keyId++, Name = name };
                                }
                            }
                        }
                    }

                    // Generate key xml
                    foreach (var keyDescription in keys.Values)
                    {
                        writer.WriteStartElement("key");
                        writer.WriteAttributeString("id", keyDescription.Id);
                        writer.WriteAttributeString("for", "node");
                        writer.WriteAttributeString("attr.name", keyDescription.Name);
                        writer.WriteAttributeString("attr.type", "string");
                        writer.WriteEndElement();
                    }

                    writer.WriteStartElement("graph");
                    writer.WriteAttributeString("id", "G");
                    writer.WriteAttributeString("edgedefault", "directed");

                    // Root
                    writer.WriteStartElement("data");
                    writer.WriteAttributeString("key", "d1");
                    writer.WriteString(set[op]);
                    writer.WriteEndElement();

                    // Cost if provided
                    if (costOp != null)
                    {
                        writer.WriteStartElement("data");
                        writer.WriteAttributeString("key", "d2");
                        writer.WriteString(set[costOp]);
                        writer.WriteEndElement();
                    }

                    foreach (var pair in set)
                    {
                        writer.WriteStartElement("node");
                        writer.WriteAttributeString("id", pair.Value);

                        writer.WriteStartElement("data");
                        writer.WriteAttributeString("key", "d0");
                        writer.WriteString(pair.Key.GetType().FullName);
                        writer.WriteEndElement();

                        var data = pair.Key.GetData();
                        if (data.Any())
                        {
                            foreach (var o in data)
                            {
                                var name = pair.Key.GetType().Name + "." + o.Key;
                                var keyDesc = keys[name];

                                writer.WriteStartElement("data");
                                writer.WriteAttributeString("key", keyDesc.Id);
                                writer.WriteString(o.Value.ToString());
                                writer.WriteEndElement();
                            }
                        }

                        writer.WriteEndElement();
                    }

                    foreach (var pair in set)
                    {
                        foreach (var valueParent in pair.Key.Parents)
                        {
                            writer.WriteStartElement("edge");
                            writer.WriteAttributeString("source", set[valueParent]);
                            writer.WriteAttributeString("target", set[pair.Key]);
                            writer.WriteEndElement();
                        }
                    }

                    writer.WriteEndElement();

                    writer.WriteEndElement();
                }

                return sw.ToString();
            }
        }

        private class KeyDescription
        {
            public string @for = "node";
            public string Id;

            public string Name;
        }

        [DebuggerDisplay("Node ({Id}:{Type})")]
        private class Node
        {
            public readonly Dictionary<string, object> Data = new Dictionary<string, object>();
            public string Id;
        }

        [DebuggerDisplay("Edge {Source}->{Target}")]
        private class Edge
        {
            public string Source;

            public string Target;
        }
    }
}