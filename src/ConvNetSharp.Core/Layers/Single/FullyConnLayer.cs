using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Single
{
    public class FullyConnLayer : FullyConnLayer<float>
    {
        public FullyConnLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public FullyConnLayer(int neuronCount) : base(neuronCount)
        {
        }
    }
}