using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Double
{
    public class FullyConnLayer : FullyConnLayer<double>
    {
        public FullyConnLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public FullyConnLayer(int neuronCount) : base(neuronCount)
        {
        }
    }
}