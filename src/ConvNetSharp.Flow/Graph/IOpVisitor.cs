using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Graph
{
    public interface IOpVisitor<T> where T : struct, IEquatable<T>, IFormattable
    {
        void Visit(Op<T> op);
    }
}