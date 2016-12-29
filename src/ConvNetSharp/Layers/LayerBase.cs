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
    public abstract class LayerBase: ISerializable
    {
        public LayerBase()
        {

        }

        public Volume InputActivation { get; set; }

        public Volume OutputActivation { get; set; }

        [DataMember]
        public int OutputDepth { get; set; }

        [DataMember]
        public int OutputWidth { get; set; }

        [DataMember]
        public int OutputHeight { get; set; }

        [DataMember]
        public int InputDepth { get; set; }

        [DataMember]
        public int InputWidth { get; set; }

        [DataMember]
        public int InputHeight { get; set; }

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

        #region Serialization

        public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("OutputDepth", this.OutputDepth, typeof(int));
            info.AddValue("OutputWidth", this.OutputWidth, typeof(int));
            info.AddValue("OutputHeight", this.OutputHeight, typeof(int));
            info.AddValue("InputDepth", this.InputDepth, typeof(int));
            info.AddValue("InputWidth", this.InputWidth, typeof(int));
            info.AddValue("InputHeight", this.InputHeight, typeof(int));
        }

        protected LayerBase(SerializationInfo info, StreamingContext context)
        {
            this.OutputDepth = info.GetInt32("OutputDepth");
            this.OutputWidth = info.GetInt32("OutputWidth");
            this.OutputHeight = info.GetInt32("OutputHeight");
            this.InputDepth = info.GetInt32("InputDepth");
            this.InputWidth = info.GetInt32("InputWidth");
            this.InputHeight = info.GetInt32("InputHeight");
        }

        #endregion
    }
}