using System.Numerics;

namespace ConvNetSharp.Flow.Ops;

public interface INamedOp<T> where T : INumber<T>, IMinMaxValue<T>
{
    string Name { get; set; }
}