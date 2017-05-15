using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Single
{
    public class ConvLayer : ConvLayer<float>
    {
        public ConvLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public ConvLayer(int width, int height, int filterCount) : base(width, height, filterCount)
        {
        }
    }
}