using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Layers
{
    public class OpWrapperLayer<T> : LayerBase<T>, ILastLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        public OpWrapperLayer(Op<T> root, Op<T> costOp)
        {
            this.Op = root;
            this.Cost = costOp;
        }

        public Op<T> Cost { get; }
    }
}