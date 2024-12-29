using System.Diagnostics;
using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     Variable hold a named Volume that can change over time.
/// </summary>
[DebuggerDisplay("{Name} (Learnable: {IsLearnable}, Shape: {Result?.Shape})")]
public class Variable<T> : Op<T>, IPersistable<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public Variable(ConvNetSharp<T> graph, Volume<T> v, string name, bool isLearnable = false) : base(graph)
    {
        this.Name = name ?? throw new ArgumentNullException(nameof(name));
        this.Result = v ?? throw new ArgumentNullException(nameof(v));
        this.IsLearnable = isLearnable;
    }

    public Variable(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
        if (!data.TryGetValue(Keys.Name, out var name) || name is not string nameStr)
        {
            throw new ArgumentException("Invalid data: 'Name' is required.");
        }

        if (!data.TryGetValue(Keys.IsLearnable, out var isLearnable) || isLearnable is not string isLearnableStr)
        {
            throw new ArgumentException("Invalid data: 'IsLearnable' is required.");
        }

        this.Name = nameStr;
        this.IsLearnable = isLearnableStr == "True";
    }

    public override string Representation => this.Name;

    /// <summary>
    ///     If set to true optimizer will try to reduce cost by updating this variable
    /// </summary>
    public bool IsLearnable { get; }

    public string Name { get; }

    public override void Differentiate()
    {
        if (this.Derivative == null)
        {
            this.Derivative = new Const<T>(this.Graph, ConvNetSharp<T>.Zero, $"{this.Name}_ZeroGradient");
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Result?.Dispose();
        }

        base.Dispose(disposing);
    }

    public override Dictionary<string, object> GetData()
    {
        var data = base.GetData();
        data[Keys.Name] = this.Name;
        data[Keys.IsLearnable] = this.IsLearnable;
        return data;
    }

    public void SetValue(Volume<T> value)
    {
        this.Result = value;
        this.SetDirty();
    }

    public override string ToString()
    {
        return this.Name;
    }

    private static class Keys
    {
        public const string Name = "Name";
        public const string IsLearnable = "IsLearnable";
    }
}