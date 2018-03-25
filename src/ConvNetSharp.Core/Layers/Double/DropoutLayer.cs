using System;
using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Double
{
    public class DropoutLayer : DropoutLayer<double>
    {
        public DropoutLayer(double dropoutProbability) : base(dropoutProbability)
        {
        }

        public DropoutLayer(Dictionary<string, object> data) : base(data)
        {
            this.DropProbability = Convert.ToDouble(data["DropProbability"]);
        }
    }
}