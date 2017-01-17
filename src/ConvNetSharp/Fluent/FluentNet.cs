using ConvNetSharp.Layers;
using System;
using System.Collections.Generic;

namespace ConvNetSharp.Fluent
{
    [Serializable]
    public class FluentNet : INet
    {
        private LastLayerBase lastLayer;

        public List<LayerBase> Layers { get; private set; } = new List<LayerBase>();

        public FluentNet(LastLayerBase layer)
        {
            this.lastLayer = layer;

            this.FindLayers(layer, this.InputLayers, this.Layers);
        }

        public List<InputLayer> InputLayers { get; private set; } = new List<InputLayer>();

        private void FindLayers(LayerBase layer, List<InputLayer> inputLayers, List<LayerBase> allLayers)
        {
            var inputLayer = layer as InputLayer;
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

            this.Layers.Add(layer);
        }

        public IVolume Forward(IVolume[] inputs, bool isTraining = false)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                this.InputLayers[i].Forward(inputs[i], isTraining);
            }

            return this.lastLayer.Forward(isTraining);
        }

        public IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputLayers[0].Forward(input, isTraining);

            return this.lastLayer.Forward(isTraining);
        }

        public double GetCostLoss(IVolume input, double y)
        {
            this.Forward(input);

            return this.lastLayer.Backward(y);
        }

        public double GetCostLoss(IVolume input, double[] y)
        {
            this.Forward(input);

            return this.lastLayer.Backward(y);
        }

        public double GetCostLoss(IVolume[] inputs, double y)
        {
            this.Forward(inputs);

            return this.lastLayer.Backward(y);
        }

        public double GetCostLoss(IVolume[] inputs, double[] y)
        {
            this.Forward(inputs);

            return this.lastLayer.Backward(y);
        }

        private void Backward(LayerBase layer)
        {
            foreach (var parent in layer.Parents)
            {
                parent.Backward();
                this.Backward(parent);
            }
        }

        public double Backward(double y)
        {
            if (this.lastLayer != null)
            {
                var loss = this.lastLayer.Backward(y); // last layer assumed to be loss layer
                this.Backward(this.lastLayer);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double Backward(double[] y)
        {
            if (this.lastLayer != null)
            {
                var loss = this.lastLayer.Backward(y); // last layer assumed to be loss layer
                this.Backward(this.lastLayer);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public int GetPrediction()
        {
            // this is a convenience function for returning the argmax
            // prediction, assuming the last layer of the net is a softmax
            var softmaxLayer = this.lastLayer as SoftmaxLayer;
            if (softmaxLayer == null)
            {
                throw new Exception("GetPrediction function assumes softmax as last layer of the net!");
            }

            var maxv = softmaxLayer.OutputActivation.Get(0);
            var maxi = 0;

            for (var i = 1; i < softmaxLayer.OutputActivation.Length; i++)
            {
                if (softmaxLayer.OutputActivation.Get(i) > maxv)
                {
                    maxv = softmaxLayer.OutputActivation.Get(i);
                    maxi = i;
                }
            }

            return maxi; // return index of the class with highest class probability
        }

        public List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();

            foreach (LayerBase t in this.Layers)
            {
                List<ParametersAndGradients> parametersAndGradients = t.GetParametersAndGradients();
                response.AddRange(parametersAndGradients);
            }

            return response;
        }

        public static LayerBase Merge(params LayerBase[] layers)
        {
            return new MergeLayer(layers);
        }

        public static InputLayer Create(int inputWidth, int inputHeight, int inputDepth)
        {
            return new InputLayer(inputWidth, inputHeight, inputDepth);
        }

        public void ReplaceLayer(LayerBase toBeReplaced, LayerBase newLayer)
        {
            for (int i = 0; i < this.Layers.Count; i++)
            {
                if (this.Layers[i] == toBeReplaced)
                {
                    foreach(var parent in this.Layers[i].Parents)
                    {
                        parent.ConnectTo(toBeReplaced);
                    }

                    toBeReplaced.ConnectTo(this.Layers[i].Child);
                    this.Layers.Clear();
                    this.InputLayers.Clear();
                    this.FindLayers(this.lastLayer, this.InputLayers, this.Layers);
                }
            }
        }
    }
}
