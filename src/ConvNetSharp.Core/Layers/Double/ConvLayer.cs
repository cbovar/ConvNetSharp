using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Double
{
    public class ConvLayer : ConvLayer<double>
    {
        public ConvLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public ConvLayer(int width, int height, int filterCount) : base(width, height, filterCount)
        {
        }
    }
}