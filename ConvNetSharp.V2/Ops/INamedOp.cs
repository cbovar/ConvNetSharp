using System.Numerics;

namespace ConvNetSharp.V2.Ops;

public interface INamedOp<T> where T : INumber<T>, IMinMaxValue<T>
{
    string Name { get; }
}