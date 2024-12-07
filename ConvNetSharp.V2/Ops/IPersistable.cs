using System.Numerics;

namespace ConvNetSharp.V2.Ops;

public interface IPersistable<T> : INamedOp<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    Volume<T> Result { get; set; }
}