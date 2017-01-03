using ConvNetSharp.Layers;
using System;
using System.Collections.Generic;

namespace ConvNetSharp.Fluent
{
    public class FluentNet : INet
    {
        private LastLayerBase lastLayer;
        List<LayerBase> allLayers = new List<LayerBase>();

        public FluentNet(LastLayerBase layer)
        {
            this.lastLayer = layer;

            this.FindLayers(layer, this.InputLayers, this.allLayers);
        }

        public List<InputLayer> InputLayers { get; private set; } = new List<InputLayer>();

        private void FindLayers(LayerBase layer, List<InputLayer> inputLayers, List<LayerBase> allLayers)
        {
            allLayers.Add(layer);

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
        }

        public IVolume Forward(bool isTraining = false, params IVolume[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                this.InputLayers[i].Forward(inputs[i], isTraining);
            }

            return this.lastLayer.Forward(isTraining);
        }

        public double GetCostLoss(IVolume volume, double y)
        {
            return 0.0;
            //this.Forward(volume);

            //var lastLayer = this.layers[this.layers.Count - 1] as ILastLayer;
            //if (lastLayer != null)
            //{
            //    var loss = lastLayer.Backward(y);
            //    return loss;
            //}

            //throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double GetCostLoss(IVolume volume, double[] y)
        {
            return 0.0;
            //this.Forward(volume);

            //var lastLayer = this.layers[this.layers.Count - 1] as ILastLayer;
            //if (lastLayer != null)
            //{
            //    var loss = lastLayer.Backward(y);
            //    return loss;
            //}

            //throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        private void Backward(LayerBase layer)
        {
            foreach(var parent in layer.Parents)
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

            var maxv = softmaxLayer.OutputActivation.GetWeight(0);
            var maxi = 0;

            for (var i = 1; i < softmaxLayer.OutputActivation.Length; i++)
            {
                if (softmaxLayer.OutputActivation.GetWeight(i) > maxv)
                {
                    maxv = softmaxLayer.OutputActivation.GetWeight(i);
                    maxi = i;
                }
            }

            return maxi; // return index of the class with highest class probability
        }

        public List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();

            foreach (LayerBase t in this.allLayers)
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
    }
}
