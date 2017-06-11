using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Double
{
    public class TanhLayer : TanhLayer<double>
    {
        public TanhLayer()
        {
        }

        public TanhLayer(Dictionary<string, object> data) : base(data)
        {
        }
    }
}