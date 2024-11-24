using System.Numerics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops;

public interface IPersistable<T> : INamedOp<T> where T : INumber<T>, IMinMaxValue<T>
{
    Volume<T> Result { get; set; }
}