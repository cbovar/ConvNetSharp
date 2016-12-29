using System;
using System.Runtime.Serialization;

namespace ConvNetSharp.Layers
{
    [DataContract]
    [Serializable]
    public sealed class InputLayer : LayerBase
    {
        public InputLayer()
        {
        }

        public InputLayer(int inputWidth, int inputHeight, int inputDepth)
        {
            this.Init(inputWidth, inputHeight, inputDepth);

            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;
            this.OutputDepth = inputDepth;
        }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
            this.InputActivation = input;
            this.OutputActivation = input;
            return this.OutputActivation; // simply identity function for now
        }

        public override void Backward()
        {
        }

        #region Serialization

        private InputLayer(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }

        #endregion
    }
}