using System;
using System.Collections.Generic;
using ConvNetSharp.Layers;

namespace ConvNetSharp
{
    [Serializable]
    public class Net
    {
        private readonly List<LayerBase> layers = new List<LayerBase>();

        public List<LayerBase> Layers
        {
            get { return this.layers; }
        }

        public void AddLayer(LayerBase layer)
        {
            int inputWidth = 0, inputHeight = 0, inputDepth = 0;
            if (this.layers.Count > 0)
            {
                inputWidth = this.layers[this.layers.Count - 1].OutputWidth;
                inputHeight = this.layers[this.layers.Count - 1].OutputHeight;
                inputDepth = this.layers[this.layers.Count - 1].OutputDepth;
            }

            var classificationLayer = layer as IClassificationLayer;
            if (classificationLayer != null)
            {
                var fullyConnLayer = new FullyConnLayer(classificationLayer.ClassCount);
                fullyConnLayer.Init(inputWidth, inputHeight, inputDepth);
                inputWidth = fullyConnLayer.OutputWidth;
                inputHeight = fullyConnLayer.OutputHeight;
                inputDepth = fullyConnLayer.OutputDepth;

                this.layers.Add(fullyConnLayer);
            }

            var regressionLayer = layer as RegressionLayer;
            if (regressionLayer != null)
            {
                var fullyConnLayer = new FullyConnLayer(regressionLayer.NeuronCount);
                fullyConnLayer.Init(inputWidth, inputHeight, inputDepth);
                inputWidth = fullyConnLayer.OutputWidth;
                inputHeight = fullyConnLayer.OutputHeight;
                inputDepth = fullyConnLayer.OutputDepth;

                this.layers.Add(fullyConnLayer);
            }

            var dotProductLayer = layer as IDotProductLayer;
            if (dotProductLayer != null)
            {
                if (dotProductLayer.Activation == Activation.Relu)
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

            if (dotProductLayer != null)
            {
                switch (dotProductLayer.Activation)
                {
                    case Activation.Undefined:
                        break;
                    case Activation.Relu:
                        var reluLayer = new ReluLayer();
                        reluLayer.Init(layer.OutputWidth, layer.OutputHeight, layer.OutputDepth);
                        this.layers.Add(reluLayer);
                        break;
                    case Activation.Sigmoid:
                        var sigmoidLayer = new SigmoidLayer();
                        sigmoidLayer.Init(layer.OutputWidth, layer.OutputHeight, layer.OutputDepth);
                        this.layers.Add(sigmoidLayer);
                        break;
                    case Activation.Tanh:
                        var tanhLayer = new TanhLayer();
                        tanhLayer.Init(layer.OutputWidth, layer.OutputHeight, layer.OutputDepth);
                        this.layers.Add(tanhLayer);
                        break;
                    case Activation.Maxout:
                        var maxoutLayer = new MaxoutLayer { GroupSize = dotProductLayer.GroupSize };
                        maxoutLayer.Init(layer.OutputWidth, layer.OutputHeight, layer.OutputDepth);
                        this.layers.Add(maxoutLayer);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }

            var lastLayer = this.layers[this.layers.Count - 1];

            if (!(layer is DropOutLayer) && layer.DropProb.HasValue)
            {
                var dropOutLayer = new DropOutLayer(layer.DropProb.Value);
                dropOutLayer.Init(lastLayer.OutputWidth, lastLayer.OutputHeight, lastLayer.OutputDepth);
                this.layers.Add(dropOutLayer);
            }
        }

        public Volume Forward(Volume volume, bool isTraining = false)
        {
            var activation = this.layers[0].Forward(volume, isTraining);

            for (var i = 1; i < this.layers.Count; i++)
            {
                var layerBase = this.layers[i];
                activation = layerBase.Forward(activation, isTraining);
            }

            return activation;
        }

        public double GetCostLoss(Volume volume, double y)
        {
            this.Forward(volume);

            var lastLayer = this.layers[this.layers.Count - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double GetCostLoss(Volume volume, double[] y)
        {
            this.Forward(volume);

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

            double[] p = softmaxLayer.OutputActivation.Weights;
            var maxv = p[0];
            var maxi = 0;

            for (var i = 1; i < p.Length; i++)
            {
                if (p[i] > maxv)
                {
                    maxv = p[i];
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