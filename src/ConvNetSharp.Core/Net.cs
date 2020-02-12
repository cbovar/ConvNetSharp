using System;
using System.Collections.Generic;
using System.IO;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core
{
    public class Net<T> : INet<T> where T : struct, IEquatable<T>, IFormattable
    {
        public List<LayerBase<T>> Layers { get; } = new List<LayerBase<T>>();

        public Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            var activation = this.Layers[0].DoForward(input, isTraining);

            for (var i = 1; i < this.Layers.Count; i++)
            {
                var layer = this.Layers[i];
                activation = layer.DoForward(activation, isTraining);
            }

            return activation;
        }

        public T GetCostLoss(Volume<T> input, Volume<T> y)
        {
            this.Forward(input);

            if (this.Layers[^1] is ILastLayer<T> lastLayer)
            {
                lastLayer.Backward(y, out var loss);
                return loss;
            }

            throw new Exception("Last layer doesn't implement ILastLayer interface");
        }

        public T Backward(Volume<T> y)
        {
            var n = this.Layers.Count;
            if (!(this.Layers[n - 1] is ILastLayer<T> lastLayer))
            {
                throw new Exception("Last layer doesn't implement ILastLayer interface");
            }

            lastLayer.Backward(y, out var loss); // last layer assumed to be loss layer
            for (var i = n - 2; i >= 0; i--)
            {
                // first layer assumed input
                this.Layers[i].Backward(this.Layers[i + 1].InputActivationGradients);
            }

            return loss;
        }

        public int[] GetPrediction()
        {
            // this is a convenience function for returning the argmax
            // prediction, assuming the last layer of the net is a softmax
            if (!(this.Layers[^1] is SoftmaxLayer<T> softmaxLayer))
            {
                throw new Exception("GetPrediction function assumes softmax as last layer of the net!");
            }

            var activation = softmaxLayer.OutputActivation;
            var N = activation.Shape.Dimensions[3];
            var C = activation.Shape.Dimensions[2];
            var result = new int[N];

            for (var n = 0; n < N; n++)
            {
                var maxv = activation.Get(0, 0, 0, n);
                var maxi = 0;

                for (var i = 1; i < C; i++)
                {
                    var output = activation.Get(0, 0, i, n);
                    if (Ops<T>.GreaterThan(output, maxv))
                    {
                        maxv = output;
                        maxi = i;
                    }
                }

                result[n] = maxi;
            }

            return result;
        }

        public List<ParametersAndGradients<T>> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients<T>>();

            foreach (var t in this.Layers)
            {
                var parametersAndGradients = t.GetParametersAndGradients();
                response.AddRange(parametersAndGradients);
            }

            return response;
        }

        public void AddLayer(LayerBase<T> layer)
        {
            int inputWidth = 0, inputHeight = 0, inputDepth = 0;
            LayerBase<T> lastLayer = null;

            if (this.Layers.Count > 0)
            {
                inputWidth = this.Layers[^1].OutputWidth;
                inputHeight = this.Layers[^1].OutputHeight;
                inputDepth = this.Layers[^1].OutputDepth;
                lastLayer = this.Layers[^1];
            }
            else if (!(layer is InputLayer<T>))
            {
                throw new ArgumentException("First layer should be an InputLayer");
            }

            if (layer is IClassificationLayer classificationLayer)
            {
                if (!(lastLayer is FullyConnLayer<T> fullconLayer))
                {
                    throw new ArgumentException(
                        $"Previously added layer should be a FullyConnLayer with {classificationLayer.ClassCount} Neurons");
                }

                if (fullconLayer.NeuronCount != classificationLayer.ClassCount)
                {
                    throw new ArgumentException(
                        $"Previous FullyConnLayer should have {classificationLayer.ClassCount} Neurons");
                }
            }

            if (layer is ReluLayer<T> || layer is LeakyReluLayer<T>)
            {
                if (lastLayer is IDotProductLayer<T> dotProductLayer)
                {
                    // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.

                    dotProductLayer.BiasPref = (T)Convert.ChangeType(0.1, typeof(T)); // can we do better?
                }
            }

            if (this.Layers.Count > 0)
            {
                layer.Init(inputWidth, inputHeight, inputDepth);
            }

            this.Layers.Add(layer);
        }

        public void Dump(string filename)
        {
            using var stream = File.Create(filename);
            using var sw = new StreamWriter(stream);

            for (var index = 0; index < this.Layers.Count; index++)
            {
                var layerBase = this.Layers[index];
                sw.WriteLine($"=== Layer {index}");
                sw.WriteLine("Input");
                sw.Write(layerBase.InputActivation.ToString());

                if (layerBase is ConvLayer<T> conv)
                {
                    sw.WriteLine("Filter");
                    sw.Write(conv.Filters.ToString());

                    sw.WriteLine("Bias");
                    sw.Write(conv.Bias.ToString());
                }

                if (layerBase is FullyConnLayer<T> full)
                {
                    sw.WriteLine("Filter");
                    sw.Write(full.Filters.ToString());

                    sw.WriteLine("Bias");
                    sw.Write(full.Bias.ToString());
                }
            }
        }

        public Volume<T> Forward(Volume<T>[] inputs, bool isTraining = false)
        {
            return this.Forward(inputs[0], isTraining);
        }

        public static Net<T> FromData(IDictionary<string, object> dico)
        {
            var net = new Net<T>();

            var layers = dico["Layers"] as IEnumerable<IDictionary<string, object>>;
            foreach (var layerData in layers)
            {
                var layer = LayerBase<T>.FromData(layerData);
                net.Layers.Add(layer);
            }

            return net;
        }

        public Dictionary<string, object> GetData()
        {
            var dico = new Dictionary<string, object>();
            var layers = new List<Dictionary<string, object>>();

            foreach (var layer in this.Layers)
            {
                layers.Add(layer.GetData());
            }

            dico["Layers"] = layers;

            return dico;
        }
    }
}