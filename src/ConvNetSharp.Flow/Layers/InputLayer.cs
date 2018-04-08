using System;

namespace ConvNetSharp.Flow.Layers
{
    public class InputLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public InputLayer()
        {
            var graph = new ConvNetSharp<T>();
            this.Op = graph.PlaceHolder("input");
        }
    }
}