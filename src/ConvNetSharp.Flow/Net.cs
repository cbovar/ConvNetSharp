using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Flow.Layers;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow
{
    public class Net<T> : INet<T> where T : struct, IEquatable<T>, IFormattable
    {
        public List<LayerBase<T>> Layers { get; } = new List<LayerBase<T>>();

        public void AddLayer(LayerBase<T> layer)
        {
            var previousLayer = this.Layers.LastOrDefault();

            if (previousLayer != null)
            {
                layer.AcceptParent(previousLayer);
            }

            this.Layers.Add(layer);
        }

        public T Backward(Volume<T> y)
        {
            throw new NotImplementedException();
        }

        public Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            throw new NotImplementedException();
        }

        public Op<T> Build()
        {
            return this.Layers.Last().Op;
        }
    }
}