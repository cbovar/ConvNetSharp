using System;
using System.Collections.Generic;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Fluent
{
    public class FluentNet<T> : INet<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly List<LayerBase<T>> _allLayers = new List<LayerBase<T>>();
        private readonly LastLayerBase<T> _lastLayer;

        public FluentNet(LastLayerBase<T> layer)
        {
            this._lastLayer = layer;

            this.FindLayers(layer, this.InputLayers, this._allLayers);
        }

        public List<InputLayer<T>> InputLayers { get; } = new List<InputLayer<T>>();

        public Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            this.InputLayers[0].DoForward(input, isTraining);

            return this._lastLayer.Forward(isTraining);
        }

        public T GetCostLoss(Volume<T> input, Volume<T> y)
        {
            this.Forward(input);

            if (this._lastLayer != null)
            {
                this._lastLayer.Backward(y, out var loss);
                return loss;
            }

            throw new Exception("Last layer doesn't implement ILastLayer interface");
        }

        public T Backward(Volume<T> y)
        {
            if (this._lastLayer != null)
            {
                this._lastLayer.Backward(y, out var loss); // last layer assumed to be loss layer
                this.Backward(this._lastLayer);
                return loss;
            }

            throw new Exception("Last layer doesn't implement ILastLayer interface");
        }

        public int[] GetPrediction()
        {
            // this is a convenience function for returning the argmax
            // prediction, assuming the last layer of the net is a softmax
            if (!(this._lastLayer is SoftmaxLayer<T> softmaxLayer))
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

            foreach (var t in this._allLayers)
            {
                var parametersAndGradients = t.GetParametersAndGradients();
                response.AddRange(parametersAndGradients);
            }

            return response;
        }

        private void Backward(LayerBase<T> layer)
        {
            foreach (var parent in layer.Parents)
            {
                parent.Backward(layer.InputActivationGradients);
                this.Backward(parent);
            }
        }

        public static InputLayer<T> Create(int inputWidth, int inputHeight, int inputDepth)
        {
            return new InputLayer<T>(inputWidth, inputHeight, inputDepth);
        }

        private void FindLayers(LayerBase<T> layer, List<InputLayer<T>> inputLayers, List<LayerBase<T>> allLayers)
        {
            allLayers.Add(layer);

            if (layer is InputLayer<T> inputLayer)
            {
                inputLayers.Add(inputLayer);
            }
            else
            {
                foreach (var parent in layer.Parents)
                {
                    this.FindLayers(parent, inputLayers, allLayers);
                }
            }
        }

        public Dictionary<string, object> GetData()
        {
            var dico = new Dictionary<string, object>();
            var layers = new List<Dictionary<string, object>>();

            for (var index = this._allLayers.Count - 1; index >= 0; index--)
            {
                var layer = this._allLayers[index];
                layers.Add(layer.GetData());
            }

            dico["Layers"] = layers;

            return dico;
        }
    }
}