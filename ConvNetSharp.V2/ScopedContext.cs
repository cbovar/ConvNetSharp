using System.Numerics;

namespace ConvNetSharp.V2;

/// <summary>
///     Manages a named scope for operations within ConvNetSharp.
///     Automatically releases the scope when disposed.
/// </summary>
/// <typeparam name="T">The numeric type used in ConvNetSharp.</typeparam>
public class ScopedContext<T> : IDisposable
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    private readonly string _name;
    private readonly IScopeManager _scopeManager;
    private bool _disposed;

    public ScopedContext(string name, IScopeManager scopeManager)
    {
        this._name = name ?? throw new ArgumentNullException(nameof(name), "Scope name cannot be null.");
        this._scopeManager = scopeManager ??
                             throw new ArgumentNullException(nameof(scopeManager), "Scope manager cannot be null.");

        this._scopeManager.RegisterScope(this._name);
    }

    public void Dispose()
    {
        if (this._disposed)
        {
            return;
        }

        this._scopeManager.ReleaseScope(this._name);
        this._disposed = true;
    }
}