using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Double
{
    public class RegressionLayer : RegressionLayer<double>
    {
        public RegressionLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public RegressionLayer()
        {
        }
    }
}