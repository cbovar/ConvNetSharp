
using System;

namespace ConvNetSharp.Layers
{
    [Serializable]
    public abstract class LastLayerBase : LayerBase, ILastLayer
    {
        public abstract double Backward(double[] y);

        public abstract double Backward(double y);
    }
}