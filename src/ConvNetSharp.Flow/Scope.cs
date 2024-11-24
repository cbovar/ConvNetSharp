using System;
using System.Numerics;

namespace ConvNetSharp.Flow;

public class Scope<T>(string name, ConvNetSharp<T> cns) : IDisposable
    where T : INumber<T>, IMinMaxValue<T>
{
    public void Dispose()
    {
        cns.ReleaseScope(name);
    }
}