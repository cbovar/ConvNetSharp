using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Single
{
    public class InputLayer : InputLayer<float>
    {
        public InputLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public InputLayer(int inputWidth, int inputHeight, int inputDepth) : base(inputWidth, inputHeight, inputDepth)
        {
        }
    }
}