using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    [KnownType(typeof(ConvLayer))]
    [KnownType(typeof(DropOutLayer))]
    [KnownType(typeof(FullyConnLayer))]
    [KnownType(typeof(InputLayer))]
    [KnownType(typeof(MaxoutLayer))]
    [KnownType(typeof(PoolLayer))]
    [KnownType(typeof(RegressionLayer))]
    [KnownType(typeof(ReluLayer))]
    [KnownType(typeof(SigmoidLayer))]
    [KnownType(typeof(SoftmaxLayer))]
    [KnownType(typeof(SvmLayer))]
    [KnownType(typeof(TanhLayer))]
    [DataContract]
    [Serializable]
    public abstract class LayerBase
    {
        public Volume InputActivation { get; protected set; }

        public Volume OutputActivation { get; protected set; }

        [DataMember]
        public int OutputDepth { get; protected set; }

        [DataMember]
        public int OutputWidth { get; protected set; }

        [DataMember]
        public int OutputHeight { get; protected set; }

        [DataMember]
        public int InputDepth { get; private set; }

        [DataMember]
        public int InputWidth { get; private set; }

        [DataMember]
        public int InputHeight { get; private set; }

        [DataMember]
        public double? DropProb { get; protected set; }

        public abstract Volume Forward(Volume input, bool isTraining = false);

        public abstract void Backward();

        public virtual void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            this.InputWidth = inputWidth;
            this.InputHeight = inputHeight;
            this.InputDepth = inputDepth;
        }

        public virtual List<ParametersAndGradients> GetParametersAndGradients()
        {
            return new List<ParametersAndGradients>();
        }
    }
}