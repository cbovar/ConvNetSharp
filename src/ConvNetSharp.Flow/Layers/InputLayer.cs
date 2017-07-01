using System;

namespace ConvNetSharp.Flow.Layers
{
    public class InputLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public InputLayer()
        {
            this.Op = ConvNetSharp<T>.Instance.PlaceHolder("input");
        }
    }
}