using System;
using System.Numerics;

namespace ConvNetSharp.Flow;

public class Scope<T> : IDisposable where T : INumber<T>, IMinMaxValue<T>
{
    private readonly ConvNetSharp<T> _cns;
    private readonly string _name;

    public Scope(string name, ConvNetSharp<T> cns)
    {
        this._name = name;
        this._cns = cns;
    }

    public void Dispose()
    {
        this._cns.ReleaseScope(this._name);
    }
}