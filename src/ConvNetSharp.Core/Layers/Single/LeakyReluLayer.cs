using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Single
{
    public class LeakyReluLayer : LeakyReluLayer<float>
    {
        public LeakyReluLayer()
        {

        }

        public LeakyReluLayer(Dictionary<string, object> data) : base(data)
        {
        }
    }
}