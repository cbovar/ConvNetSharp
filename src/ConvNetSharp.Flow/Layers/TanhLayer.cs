using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers
{
    public class TanhLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public override void AcceptParent(Op<T> parent)
        {
            this.Op = ConvNetSharp<T>.Tanh(parent);
        }
    }
}