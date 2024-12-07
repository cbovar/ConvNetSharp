namespace ConvNetSharp.V2;

public interface IScopeManager
{
    void RegisterScope(string name);
    void ReleaseScope(string name);
}