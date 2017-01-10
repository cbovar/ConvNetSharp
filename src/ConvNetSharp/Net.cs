using System;
using System.Collections.Generic;
using ConvNetSharp.Layers;

namespace ConvNetSharp
{
    [Serializable]
    public class Net : INet
    {
        private readonly List<LayerBase> layers = new List<LayerBase>();

        public List<LayerBase> Layers
        {
            get { return this.layers; }
        }

        public void AddLayer(LayerBase layer)
        {
            int inputWidth = 0, inputHeight = 0, inputDepth = 0;
            LayerBase lastLayer = null;

            if (this.layers.Count > 0)
            {
                inputWidth = this.layers[this.layers.Count - 1].OutputWidth;
                inputHeight = this.layers[this.layers.Count - 1].OutputHeight;
                inputDepth = this.layers[this.layers.Count - 1].OutputDepth;
                lastLayer = this.layers[this.layers.Count - 1];
            }
            else if (!(layer is InputLayer))
            {
                throw new ArgumentException("First layer should be an InputLayer");
            }

            var classificationLayer = layer as IClassificationLayer;
            if (classificationLayer != null)
            {
                var fullconLayer = lastLayer as FullyConnLayer;
                if (fullconLayer == null)
                {
                    throw new ArgumentException($"Previously added layer should be a FullyConnLayer with {classificationLayer.ClassCount} Neurons");
                }

                if (fullconLayer.NeuronCount != classificationLayer.ClassCount)
                {
                    throw new ArgumentException($"Previous FullyConnLayer should have {classificationLayer.ClassCount} Neurons");
                }
            }

            var regressionLayer = layer as RegressionLayer;
            if (regressionLayer != null)
            {
                var fullconLayer = lastLayer as FullyConnLayer;
                if (fullconLayer == null)
                {
                    throw new ArgumentException("Previously added layer should be a FullyConnLayer");
                }
            }

            var reluLayer = layer as ReluLayer;
            if (reluLayer != null)
            {
                var dotProductLayer = lastLayer as IDotProductLayer;
                if (dotProductLayer != null)
                {
                    dotProductLayer.BiasPref = 0.1; // relus like a bit of positive bias to get gradients early
                                                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                                                    // and will never get any gradient and never contribute any computation. Dead relu.
                }
            }

            if (this.layers.Count > 0)
            {
                layer.Init(inputWidth, inputHeight, inputDepth);
            }

            this.layers.Add(layer);
        }

        public IVolume Forward(IVolume[] inputs, bool isTraining = false)
        {
            return this.Forward(inputs[0], isTraining);
        }

        public IVolume Forward(IVolume input, bool isTraining = false)
        {
            var activation = this.layers[0].Forward(input, isTraining);

            for (var i = 1; i < this.layers.Count; i++)
            {
                var layerBase = this.layers[i];
                activation = layerBase.Forward(activation, isTraining);
            }

            return activation;
        }

        public double GetCostLoss(IVolume input, double y)
        {
            this.Forward(input);

            var lastLayer = this.layers[this.layers.Count - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double GetCostLoss(IVolume input, double[] y)
        {
            this.Forward(input);

            var lastLayer = this.layers[this.layers.Count - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double Backward(double y)
        {
            var n = this.layers.Count;
            var lastLayer = this.layers[n - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y); // last layer assumed to be loss layer
                for (var i = n - 2; i >= 0; i--)
                {
                    // first layer assumed input
                    this.layers[i].Backward();
                }
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double Backward(double[] y)
        {
            var n = this.layers.Count;
            var lastLayer = this.layers[n - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y); // last layer assumed to be loss layer
                for (var i = n - 2; i >= 0; i--)
                {
                    // first layer assumed input
                    this.layers[i].Backward();
                }
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public int GetPrediction()
        {
            // this is a convenience function for returning the argmax
            // prediction, assuming the last layer of the net is a softmax
            var softmaxLayer = this.layers[this.layers.Count - 1] as SoftmaxLayer;
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

            foreach (LayerBase t in this.layers)
            {
                List<ParametersAndGradients> parametersAndGradients = t.GetParametersAndGradients();
                response.AddRange(parametersAndGradients);
            }

            return response;
        }
    }
}