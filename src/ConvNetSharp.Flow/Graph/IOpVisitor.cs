using System.Numerics;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Graph;

public interface IOpVisitor<T> where T : INumber<T>, IMinMaxValue<T>
{
    void Visit(Op<T> op);
}