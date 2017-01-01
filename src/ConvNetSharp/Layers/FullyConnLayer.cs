using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace ConvNetSharp.Layers
{
    [DataContract]
    [Serializable]
    public class FullyConnLayer : LayerBase, IDotProductLayer
    {
        [DataMember]
        private int inputCount;

        public FullyConnLayer(int neuronCount, Activation activation = Activation.Undefined)
        {
            this.NeuronCount = neuronCount;
            Activation = activation;

            this.L1DecayMul = 0.0;
            this.L2DecayMul = 1.0;
        }

        [DataMember]
        public Volume Biases { get; private set; }

        [DataMember]
        public List<Volume> Filters { get; private set; }

        [DataMember]
        public double L1DecayMul { get; set; }

        [DataMember]
        public double L2DecayMul { get; set; }

        [DataMember]
        public int NeuronCount { get; private set; }

        [DataMember]
        public int GroupSize { get; set; }

        [DataMember]
        public Activation Activation { get; set; }

        [DataMember]
        public double BiasPref { get; set; }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var outputActivation = new Volume(1, 1, this.OutputDepth, 0.0);
            double[] vw = input.Weights;

#if PARALLEL
            Parallel.For(0, this.OutputDepth, i =>
#else
            for (var i = 0; i < this.OutputDepth; i++)
#endif
            {
                var a = 0.0;
                double[] wi = this.Filters[i].Weights;

                for (var d = 0; d < this.inputCount; d++)
                {
                    a += vw[d] * wi[d]; // for efficiency use Vols directly for now
                }

                a += this.Biases.Weights[i];
                outputActivation.Weights[i] = a;
            }
#if PARALLEL
                );
#endif

            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation;
            volume.WeightGradients = new double[volume.Weights.Length]; // zero out the gradient in input Vol

            // compute gradient wrt weights and data
#if PARALLEL
            var lockObject = new object();
            Parallel.For(0, this.OutputDepth, () => new double[volume.Weights.Length], (i, state, temp) =>
#else
            var temp = volume.WeightGradients;
            for (var i = 0; i < this.OutputDepth; i++)
#endif
            {
                var tfi = this.Filters[i];
                var chainGradient = this.OutputActivation.WeightGradients[i];
                for (var d = 0; d < this.inputCount; d++)
                {
                    temp[d] += tfi.Weights[d] * chainGradient; // grad wrt input data
                    tfi.WeightGradients[d] += volume.Weights[d] * chainGradient; // grad wrt params
                }
                this.Biases.WeightGradients[i] += chainGradient;

#if !PARALLEL
            }
#else
                return temp;
            }
                , result =>
                {
                    lock (lockObject)
                    {
                        for (var i = 0; i < this.inputCount; i++)
                        {
                            volume.WeightGradients[i] += result[i];
                        }
                    }
                }
                );
#endif
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
            this.Filters = new List<Volume>();

            for (var i = 0; i < this.OutputDepth; i++)
            {
                this.Filters.Add(new Volume(1, 1, this.inputCount));
            }

            this.Biases = new Volume(1, 1, this.OutputDepth, bias);
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();
            for (var i = 0; i < this.OutputDepth; i++)
            {
                response.Add(new ParametersAndGradients
                {
                    Parameters = this.Filters[i].Weights,
                    Gradients = this.Filters[i].WeightGradients,
                    L2DecayMul = this.L2DecayMul,
                    L1DecayMul = this.L1DecayMul
                });
            }

            response.Add(new ParametersAndGradients
            {
                Parameters = this.Biases.Weights,
                Gradients = this.Biases.WeightGradients,
                L1DecayMul = 0.0,
                L2DecayMul = 0.0
            });

            return response;
        }
    }
}