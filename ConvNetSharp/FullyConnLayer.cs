using System.Collections.Generic;

namespace ConvNetSharp
{
    public class FullyConnLayer : LayerBase, IConvLayer
    {
        private Volume biases;
        private List<Volume> filters;
        private int inputCount;

        public FullyConnLayer(int neuronCount, Activation activation = Activation.Undefined)
        {
            this.NeuronCount = neuronCount;
            this.Activation = activation;

            this.L1DecayMul = 0.0;
            this.L2DecayMul = 1.0;
        }

        public double L1DecayMul { get; set; }

        public double L2DecayMul { get; set; }

        public int NeuronCount { get; private set; }

        public int GroupSize { get; set; }

        public Activation Activation { get; private set; }

        public double BiasPref { get; set; }

        public override Volume Forward(Volume volume, bool isTraining = false)
        {
            this.InputActivation = volume;
            var outputActivation = new Volume(1, 1, this.OutputDepth, 0.0);
            double[] vw = volume.Weights;

            for (var i = 0; i < this.OutputDepth; i++)
            {
                var a = 0.0;
                double[] wi = this.filters[i].Weights;

                for (var d = 0; d < this.inputCount; d++)
                {
                    a += vw[d] * wi[d]; // for efficiency use Vols directly for now
                }

                a += this.biases.Weights[i];
                outputActivation.Weights[i] = a;
            }

            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation;
            volume.WeightGradients = new double[volume.Weights.Length]; // zero out the gradient in input Vol

            // compute gradient wrt weights and data
            for (var i = 0; i < this.OutputDepth; i++)
            {
                var tfi = this.filters[i];
                var chainGradient = this.OutputActivation.WeightGradients[i];
                for (var d = 0; d < this.inputCount; d++)
                {
                    volume.WeightGradients[d] += tfi.Weights[d] * chainGradient; // grad wrt input data
                    tfi.WeightGradients[d] += volume.Weights[d] * chainGradient; // grad wrt params
                }
                this.biases.WeightGradients[i] += chainGradient;
            }
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // required
            // ok fine we will allow 'filters' as the word as well
            this.OutputDepth = this.NeuronCount;

            // computed
            this.inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputWidth = 1;
            this.OutputHeight = 1;

            // initializations
            var bias = this.BiasPref;
            this.filters = new List<Volume>();

            for (var i = 0; i < this.OutputDepth; i++)
            {
                this.filters.Add(new Volume(1, 1, this.inputCount));
            }

            this.biases = new Volume(1, 1, this.OutputDepth, bias);
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();
            for (var i = 0; i < this.OutputDepth; i++)
            {
                response.Add(new ParametersAndGradients
                {
                    Parameters = this.filters[i].Weights,
                    Gradients = this.filters[i].WeightGradients,
                    L2DecayMul = this.L2DecayMul,
                    L1DecayMul = this.L1DecayMul
                });
            }

            response.Add(new ParametersAndGradients
            {
                Parameters = this.biases.Weights,
                Gradients = this.biases.WeightGradients,
                L1DecayMul = 0.0,
                L2DecayMul = 0.0
            });
            return response;
        }
    }
}