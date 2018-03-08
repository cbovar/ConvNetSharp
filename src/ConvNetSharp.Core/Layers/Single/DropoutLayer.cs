using System;
using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Single
{
    public class DropoutLayer : DropoutLayer<float>
    {
        public DropoutLayer()
        {
        }

        public DropoutLayer(Dictionary<string, object> data) : base(data)
        {
            this.DropProbability = Convert.ToSingle(data["DropProbability"]);
        }
    }
}