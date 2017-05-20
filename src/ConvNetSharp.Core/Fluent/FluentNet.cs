using System;
using System.Collections.Generic;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Fluent
{
    public class FluentNet<T> : INet<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly LastLayerBase<T> _lastLayer;
        readonly List<LayerBase<T>> _allLayers = new List<LayerBase<T>>();

        public FluentNet(LastLayerBase<T> layer)
        {
            this._lastLayer = layer;

            this.FindLayers(layer, this.InputLayers, this._allLayers);
        }

        public List<InputLayer<T>> InputLayers { get; private set; } = new List<InputLayer<T>>();

        private void FindLayers(LayerBase<T> layer, List<InputLayer<T>> inputLayers, List<LayerBase<T>> allLayers)
        {
            allLayers.Add(layer);

            var inputLayer = layer as InputLayer<T>;
            if (inputLayer != null)
            {
                inputLayers.Add(inputLayer);
                return;
            }
            else
            {
                foreach (var parent in layer.Parents)
                {
                    this.FindLayers(parent, inputLayers, allLayers);
                }
            }
        }

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
                T loss;
                this._lastLayer.Backward(y, out loss);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        private void Backward(LayerBase<T> layer)
        {
            foreach (var parent in layer.Parents)
            {
                parent.Backward(layer.InputActivationGradients);
                this.Backward(parent);
            }
        }

        public T Backward(Volume<T> y)
        {
            if (this._lastLayer != null)
            {
                T loss;
                this._lastLayer.Backward(y, out loss);  // last layer assumed to be loss layer
                this.Backward(this._lastLayer);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public int[] GetPrediction()
        {
            // this is a convenience function for returning the argmax
            // prediction, assuming the last layer of the net is a softmax
            var softmaxLayer = this._lastLayer as SoftmaxLayer<T>;
            if (softmaxLayer == null)
            {
                throw new Exception("GetPrediction function assumes softmax as last layer of the net!");
            }

            var activation = softmaxLayer.OutputActivation;
            var N = activation.Shape.GetDimension(3);
            var C = activation.Shape.GetDimension(2);
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

        //public static LayerBase<T> Merge(params LayerBase<T>[] layers)
        //{
        //    return new MergeLayer(layers);
        //}

        public static InputLayer<T> Create(int inputWidth, int inputHeight, int inputDepth)
        {
            return new InputLayer<T>(inputWidth, inputHeight, inputDepth);
        }
    }
}