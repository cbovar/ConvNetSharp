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
        public IVolume InputActivation { get; protected set; }

        public IVolume OutputActivation { get; protected set; }

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
        public LayerBase Child { get; set; }

        [DataMember]
        public List<LayerBase> Parents { get; set; } = new List<LayerBase>();

        public abstract IVolume Forward(IVolume input, bool isTraining = false);

        public virtual IVolume Forward(bool isTraining)
        {
            return this.Forward(this.Parents[0].Forward(isTraining), isTraining);
        }

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

        internal void ConnectTo(LayerBase layer)
        {
            this.Child = layer;
            layer.Parents.Add(this);

            layer.Init(this.OutputWidth, this.OutputHeight, this.OutputDepth);
        }
    }
}
