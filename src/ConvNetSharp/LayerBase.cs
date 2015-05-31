using System.Collections.Generic;

namespace ConvNetSharp
{
    public class ParametersAndGradients
    {
        public double[] Parameters { get; set; }

        public double[] Gradients { get; set; }

        public double? L2DecayMul { get; set; }

        public double? L1DecayMul { get; set; }
    }

    public abstract class LayerBase
    {
        public Volume InputActivation { get; protected set; }

        public Volume OutputActivation { get; protected set; }

        public int OutputDepth { get; protected set; }

        public int OutputWidth { get; protected set; }

        public int OutputHeight { get; protected set; }

        protected int InputDepth { get; private set; }

        protected int InputWidth { get; private set; }

        protected int InputHeight { get; private set; }

        protected int Width { get; set; }

        protected int Height { get; set; }

        public double? DropProb { get; protected set; }

        public abstract Volume Forward(Volume volume, bool isTraining = false);
      
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