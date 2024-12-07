using System.Diagnostics;
using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     Const holds a Volume that will not change over time.
/// </summary>
/// <typeparam name="T"></typeparam>
[DebuggerDisplay("{Name} (Shape: {OutputShape?.ToString() ?? \"Scalar\"})")]
public class Const<T> : Op<T>, IPersistable<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public Const(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
        if (!data.TryGetValue("Name", out var nameObj) || nameObj is not string name)
        {
            throw new ArgumentException("Invalid data: 'Name' is required.");
        }

        if (!data.TryGetValue("x", out var valueObj))
        {
            throw new ArgumentException("Invalid data: 'x' (constant value) is required.");
        }

        this.Name = name;
        this.Value = (T)Convert.ChangeType(valueObj, typeof(T));

        if (data.ContainsKey("dim0"))
        {
            this.OutputShape = new Shape(
                int.Parse(data["dim0"].ToString()),
                int.Parse(data["dim1"].ToString()),
                int.Parse(data["dim2"].ToString()),
                int.Parse(data["dim3"].ToString()));
        }
    }

    public Const(ConvNetSharp<T> graph, Volume<T> v, string name) : base(graph)
    {
        this.Name = name ?? throw new ArgumentNullException(nameof(name));
        this.Result = v ?? throw new ArgumentNullException(nameof(v));
        this.OutputShape = this.Result.Shape;
    }

    public Const(ConvNetSharp<T> graph, T x, string name) : base(graph)
    {
        this.Name = name ?? throw new ArgumentNullException(nameof(name));
        this.Value = x;
    }

    public T Value { get; }

    public override string Representation => this.Name;

    public Shape OutputShape { get; }

    public string Name { get; protected set; }

    public override void Differentiate()
    {
        // No differentiation required for constant values.
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Result?.Dispose();
        }

        base.Dispose(disposing);
    }

    public override Volume<T> Evaluate(Session<T> session)
    {
        if (this.Result != null)
        {
            return base.Evaluate(session);
        }

        // Create volume with scalar or full shape
        this.Result = this.OutputShape == null
            ? Volume<T>.Create([this.Value], new Shape(1, 1, 1, 1))
            : Volume<T>.Create(Enumerable.Repeat(this.Value, (int)this.OutputShape.TotalLength).ToArray(),
                this.OutputShape);

        return base.Evaluate(session);
    }

    public override Dictionary<string, object> GetData()
    {
        var data = base.GetData();
        data["Name"] = this.Name;
        data["x"] = this.Value;

        if (this.OutputShape != null)
        {
            data["dim0"] = this.OutputShape.Dimensions[0];
            data["dim1"] = this.OutputShape.Dimensions[1];
            data["dim2"] = this.OutputShape.Dimensions[2];
            data["dim3"] = this.OutputShape.Dimensions[3];
        }

        return data;
    }

    public override string ToString()
    {
        return this.Name;
    }
}