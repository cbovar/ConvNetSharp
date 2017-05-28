using System;
using ConvNetSharp.Core.Ops;

namespace ConvNetSharp.Core.Graph
{
    public interface IOpVisitor<T> where T : struct, IEquatable<T>, IFormattable
    {
        void Visit(Op<T> op);
    }
}