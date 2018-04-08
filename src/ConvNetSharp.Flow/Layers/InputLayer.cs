using System;

namespace ConvNetSharp.Flow.Layers
{
    public class InputLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public InputLayer()
        {
            var cns = new ConvNetSharp<T>();
            this.Op = cns.PlaceHolder("input");
        }
    }
}