using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Xml;
using System.Xml.Serialization;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Serialization
{
    public static class SerializationExtensions
    {
        public static Op<T> FromXml<T>(string xml) where T : struct, IEquatable<T>, IFormattable
        {
            using (var sw = new StringReader(xml))
            {
                string root = "";
                var keys = new Dictionary<string, KeyDescription>();
                var nodes = new Dictionary<string, Node>();
                var edges = new List<Edge>();

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
                                    }
                                    break;
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

                return ops[root];
            }
        }

        public static string ToXml<T>(this Op<T> op) where T : struct, IEquatable<T>, IFormattable
        {
            var id = 0;
            var set = new Dictionary<Op<T>, string>();
            var keys = new Dictionary<string, KeyDescription>
            {
                { "d0", new KeyDescription { Id = "d0", Name = "type" } },
                { "d1", new KeyDescription { Id = "d1", Name = "root", @for = "graph"} }
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

            using (var sw = new StringWriter())
            {
                using (var writer = XmlWriter.Create(sw, new XmlWriterSettings { NewLineOnAttributes = true, Indent = true }))
                {
                    var ns = "http://graphml.graphdrawing.org/xmlns";

                    writer.WriteStartDocument();
                    writer.WriteStartElement("graphml", ns);
                    writer.WriteAttributeString("xmlns", ns);

                    // Get all keys
                    int keyId = 2;
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
            public string Id;

            public string Name;

            public string @for = "node";
        }

        private class KeyValue
        {
            public string Id;

            public object Value;
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