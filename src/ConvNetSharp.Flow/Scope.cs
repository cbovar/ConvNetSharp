using System;

namespace ConvNetSharp.Flow
{
    public class Scope<T> : IDisposable where T : struct, IEquatable<T>, IFormattable
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
}