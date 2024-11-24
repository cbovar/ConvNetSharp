using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Double;

public class DropoutLayer : DropoutLayer<double>
{
    public DropoutLayer(float dropoutProbability) : base(dropoutProbability)
    {
    }

    public DropoutLayer(Dictionary<string, object> data) : base(data)
    {
        this.DropProbability = (float)data["DropProbability"];
    }
}