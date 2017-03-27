using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Single
{
    public class SigmoidLayer : SigmoidLayer<float>
    {
        public SigmoidLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public SigmoidLayer()
        {
        }
    }
}