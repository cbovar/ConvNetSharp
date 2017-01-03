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

        public FullyConnLayer(int neuronCount)
        {
            this.NeuronCount = neuronCount;

            this.L1DecayMul = 0.0;
            this.L2DecayMul = 1.0;
        }

        [DataMember]
        public IVolume Biases { get; private set; }

        [DataMember]
        public List<IVolume> Filters { get; private set; }

        [DataMember]
        public double L1DecayMul { get; set; }

        [DataMember]
        public double L2DecayMul { get; set; }

        [DataMember]
        public int NeuronCount { get; private set; }

        [DataMember]
        public double BiasPref { get; set; }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;
            var outputActivation = new Volume(1, 1, this.OutputDepth, 0.0);

#if PARALLEL
            Parallel.For(0, this.OutputDepth, (int i) =>
#else
            for (var i = 0; i < this.OutputDepth; i++)
#endif
            {
                var a = 0.0;
                for (var d = 0; d < this.inputCount; d++)
                {
                    a += input.GetWeight(d) * this.Filters[i].GetWeight(d); // for efficiency use Vols directly for now
                }

                a += this.Biases.GetWeight(i);
                outputActivation.SetWeight(i, a);
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
            volume.ZeroGradients(); // zero out the gradient in input Vol

            // compute gradient wrt weights and data
#if PARALLEL
            var lockObject = new object();
            Parallel.For(0, this.OutputDepth, () => new double[volume.Length], (int i, ParallelLoopState state, double[] temp) =>
#else
            var temp = volume.WeightGradients;
            for (var i = 0; i < this.OutputDepth; i++)
#endif
            {
                var tfi = this.Filters[i];
                var chainGradient = this.OutputActivation.GetWeightGradient(i);
                for (var d = 0; d < this.inputCount; d++)
                {
                    temp[d] += tfi.GetWeight(d) * chainGradient; // grad wrt input data
                    tfi.SetWeightGradient(d, tfi.GetWeightGradient(d) + volume.GetWeight(d) * chainGradient); // grad wrt params
                }
                this.Biases.SetWeightGradient(i, this.Biases.GetWeightGradient(i) + chainGradient);

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
                            volume.SetWeightGradient(i, volume.GetWeightGradient(i) + result[i]);
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
            this.Filters = new List<IVolume>();

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
                    //Parameters = this.Filters[i].Weights,
                    //Gradients = this.Filters[i].WeightGradients,
                    Volume = this.Filters[i],
                    L2DecayMul = this.L2DecayMul,
                    L1DecayMul = this.L1DecayMul
                });
            }

            response.Add(new ParametersAndGradients
            {
                //Parameters = this.Biases.Weights,
                //Gradients = this.Biases.WeightGradients,
                Volume = this.Biases,
                L1DecayMul = 0.0,
                L2DecayMul = 0.0
            });

            return response;
        }
    }
}