using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Single
{
    public class TanhLayer : TanhLayer<float>
    {
        public TanhLayer()
        {
        }

        public TanhLayer(Dictionary<string, object> data) : base(data)
        {
        }
    }
}