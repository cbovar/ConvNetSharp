using System.Numerics;

namespace ConvNetSharp.V2.Ops;

public interface IOpVisitor<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    void Visit(Op<T> op);
}