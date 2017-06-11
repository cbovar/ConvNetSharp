using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public abstract class LastLayerBase<T> : LayerBase<T>, ILastLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected LastLayerBase()
        {
        }

        protected LastLayerBase(Dictionary<string, object> data) : base(data)
        {
        }

        public abstract void Backward(Volume<T> y, out T loss);
    }
}