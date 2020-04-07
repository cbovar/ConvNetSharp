using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Xml;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using Onnx;
using ProtoBuf;

namespace ConvNetSharp.Flow.Serialization.Onnx
{
    public class OnnxLoader<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly ConvNetSharp<T> _cns;
        private readonly Dictionary<string, TensorProto> _initializers;
        private readonly Dictionary<string, NodeProto> _nodes;

        public OnnxLoader(string filename, ConvNetSharp<T> cns = null)
        {
            this._cns = cns ?? new ConvNetSharp<T>();
            using var file = File.OpenRead(filename);
            var model = Serializer.Deserialize<ModelProto>(file);

            this._initializers = model.Graph.Initializers.ToDictionary(o => o.Name, o => o);

            this._nodes = new Dictionary<string, NodeProto>();
            foreach (var node in model.Graph.Nodes)
            {
                foreach (var output in node.Outputs)
                {
                    this._nodes.Add(output, node);
                }
            }
        }

        public Dictionary<string, Op<T>> Load()
        {
            var opDico = new Dictionary<string, Op<T>>();

            foreach (var (name, node) in this._nodes)
            {
                this.GetOp(name, node, opDico);
            }

            return opDico;
        }

        private Op<T> GetOp(string name, Dictionary<string, Op<T>> seen)
        {
            if (seen.TryGetValue(name, out var op))
            {
                return op;
            }

            if (this._nodes.TryGetValue(name, out var node))
            {
                op = this.GetOp(name, node, seen);
                return op;
            }

            if (this._initializers.TryGetValue(name, out var initializer))
            {
                var volume = BuildVolume(initializer);
                op = this._cns.Variable(volume, name, true);
                seen[name] = op;
                return op;
            }

            op = this._cns.PlaceHolder(name);
            seen[name] = op;
            return op;
        }

        private Op<T> GetOp(string name, NodeProto node, Dictionary<string, Op<T>> seen)
        {
            Op<T> op;

            switch (node.OpType)
            {
                case "Gemm":
                    {
                        op = this.GetGemmOp(node, seen);
                    }
                    break;
                case "Conv":
                    {
                        op = this.GetConvOp(node, seen);
                    }
                    break;
                case "Relu":
                    {
                        op = this.GetReluOp(node, seen);
                    }
                    break;
                case "MaxPool":
                    {
                        op = this.GetMaxPoolOp(node, seen);
                    }
                    break;
                default:
                    throw new Exception($"Op '{node.OpType}' not implemented in ConvNetSharp");
            }

            seen[name] = op;
            return op;
        }

        private Op<T> GetReluOp(NodeProto node, Dictionary<string, Op<T>> seen)
        {
            var input = this.GetOp(node.Inputs[0], seen);

            Op<T> op = new Activation<T>(this._cns, new Dictionary<string, object> { { "ActivationType", ActivationType.Relu.ToString() } });
            op.AddParent(input);
            return op;
        }

        private Op<T> GetMaxPoolOp(NodeProto node, Dictionary<string, Op<T>> seen)
        {
            var input = this.GetOp(node.Inputs[0], seen);

            var data = new Dictionary<string, object>
            {
                ["Width"] = node.Attributes.First(o => o.Name == "kernel_shape").Ints[0].ToString(),
                ["Height"] = node.Attributes.First(o => o.Name == "kernel_shape").Ints[1].ToString(),
                ["HorizontalPad"] = node.Attributes.First(o => o.Name == "pads").Ints[0].ToString(), // TODO: use all pads
                ["VerticalPad"] = node.Attributes.First(o => o.Name == "pads").Ints[2].ToString(),// TODO: use all pads
                ["HorizontalStride"] = node.Attributes.First(o => o.Name == "strides").Ints[0].ToString(),
                ["VerticalStride"] = node.Attributes.First(o => o.Name == "strides").Ints[1].ToString(),
            };

            Op<T> op = new Pool<T>(this._cns, data);
            op.AddParent(input);
            return op;
        }

        private Op<T> GetConvOp(NodeProto node, Dictionary<string, Op<T>> seen)
        {
            var input = this.GetOp(node.Inputs[0], seen);

            var weightVolume = BuildVolume(this._initializers[node.Inputs[1]]);
            var biasVolume = BuildVolume(this._initializers[node.Inputs[2]]);
            var weight = this._cns.Variable(weightVolume, $"Filter_{weightVolume.Shape.Dimensions[3]}", true);
            var bias = this._cns.Variable(biasVolume, "Bias", true);

            var data = new Dictionary<string, object>
            {
                ["Width"] = node.Attributes.First(o => o.Name == "kernel_shape").Ints[0].ToString(),
                ["Height"] = node.Attributes.First(o => o.Name == "kernel_shape").Ints[1].ToString(),
                ["Pad"] = node.Attributes.First(o => o.Name == "pads").Ints[0].ToString(), // TODO: use all pads
                ["Stride"] = node.Attributes.First(o => o.Name == "strides").Ints[0].ToString(), // TODO: use all strides
                ["FilterCount"] = weightVolume.Shape.Dimensions[3].ToString()
            };

            var conv = new Convolution<T>(this._cns, data);
            conv.AddParent(input);
            conv.AddParent(weight);

            var op = conv + bias;
            return op;
        }

        private Op<T> GetGemmOp(NodeProto node, Dictionary<string, Op<T>> seen)
        {
            var input = this.GetOp(node.Inputs[0], seen);

            var weightVolume = BuildVolume(this._initializers[node.Inputs[1]]);
            var biasVolume = BuildVolume(this._initializers[node.Inputs[2]]);
            var weight = this._cns.Variable(weightVolume, $"Filter_{weightVolume.Shape.Dimensions[3]}", true);
            var bias = this._cns.Variable(biasVolume, "Bias", true);

            var data = new Dictionary<string, object>
            {
                ["Alpha"] = node.Attributes.First(o => o.Name == "alpha").F, // TODO: use alpha
                ["Beta"] = node.Attributes.First(o => o.Name == "beta").F, // TODO: use beta
                // ["TransA"] = node.Attributes.First(o => o.Name == "transA").I.ToString(), // TODO: use transA
                ["TransB"] = node.Attributes.First(o => o.Name == "transB").I.ToString(), // TODO: use transB
            };

            var matMult = new MatMult<T>(this._cns, data);
            matMult.AddParent(input);
            matMult.AddParent(weight);

            var op = matMult + bias;
            return op;
        }

        /// <summary>
        ///     Make sure tensor type matches generic type T
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns>True if tensor type matches generic type T. False otherwise</returns>
        private static bool CheckType(TensorProto tensor)
        {
            switch ((TensorProto.DataType)tensor.data_type)
            {
                case TensorProto.DataType.Float:
                    {
                        if (typeof(T) == typeof(float))
                        {
                            return true;
                        }

                        break;
                    }
                case TensorProto.DataType.Double:
                    {
                        if (typeof(T) == typeof(double))
                        {
                            return true;
                        }

                        break;
                    }
            }

            return false;
        }

        /// <summary>
        ///     Make sure tensor type matches generic type T. Raise an exception if it doesn't
        /// </summary>
        /// <param name="tensor"></param>
        private static void CheckTypeOrRaiseException(TensorProto tensor)
        {
            if (!CheckType(tensor))
            {
                throw new Exception($"Incompatible types: trying to load {(TensorProto.DataType)tensor.data_type} as {typeof(T).Name}");
            }
        }

        private static Volume<T> BuildVolume(TensorProto tensor)
        {
            CheckTypeOrRaiseException(tensor);

            var shape = Shape.From(tensor.Dims.Select(o => (int)o).ToArray(), Shape.DimensionOrder.NCWH);
            var span = new Span<byte>(tensor.RawData);
            var data = MemoryMarshal.Cast<byte, T>(span);

            return BuilderInstance<T>.Volume.From(data.ToArray(), shape);
        }
    }
}