using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Double
{
    public class LeakyReluLayer : LeakyReluLayer<double>
    {
        public LeakyReluLayer(double alpha) : base(alpha)
        {
        }

        public LeakyReluLayer(Dictionary<string, object> data) : base(data)
        {
        }
    }
}