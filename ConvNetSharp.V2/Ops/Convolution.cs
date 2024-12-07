using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     y = conv(x)
/// </summary>
/// <typeparam name="T"></typeparam>
public class Convolution<T> : Op<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    private readonly Variable<T> _filter;
    private long _lastGradientComputeStep = -1;
    private Shape _lastInputShape;

    public Convolution(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
    {
        this.Stride = int.Parse((string)data["Stride"]);
        this.Pad = int.Parse((string)data["Pad"]);
        this.FilterCount = int.Parse((string)data["FilterCount"]);
        this.Width = int.Parse((string)data["Width"]);
        this.Height = int.Parse((string)data["Height"]);

        if (!data.TryGetValue("Filter", out var filter))
        {
            throw new ArgumentException("Filter must be explicitly initialized.");
        }

        this._filter = (Variable<T>)filter;
    }

    public Convolution(ConvNetSharp<T> graph, Op<T> x, Volume<T> filter, int width, int height, int filterCount,
        int stride = 1, int pad = 0)
        : base(graph)
    {
        if (filter == null)
        {
            throw new ArgumentNullException(nameof(filter), "Filter volume cannot be null.");
        }

        this.Stride = stride;
        this.Pad = pad;
        this.FilterCount = filterCount;
        this.Width = width;
        this.Height = height;

        this.AddParent(x);

        // Defer shape validation until evaluation
        this._filter = graph.Variable(filter, $"Filter_{Count}", true);
        this.AddParent(this._filter);
    }

    public Op<T> Filter => this._filter;

    public int Stride { get; set; }

    public int Pad { get; set; }

    public int FilterCount { get; }

    public int Width { get; }

    public int Height { get; }

    public override string Representation => this.Width == 1 && this.Height == 1
        ? "Dense"
        : $"Conv {this.Width}x{this.Height}x{this.FilterCount}";

    public Volume<T> FilterGradient { get; private set; }

    public Volume<T> InputGradient { get; private set; }

    public override void Differentiate()
    {
        this.Parents[0].RegisterDerivative(new ConvolutionInputGradient<T>(this.Graph, this, this.Derivative));
        this.Parents[1].RegisterDerivative(new ConvolutionFilterGradient<T>(this.Graph, this, this.Derivative));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Result?.Dispose();
            this.Parents[1]?.Dispose();
            this.FilterGradient?.Dispose();
            this.InputGradient?.Dispose();
        }

        base.Dispose(disposing);
    }

    public override Volume<T> Evaluate(Session<T> session)
    {
        if (!this.IsDirty)
        {
            return base.Evaluate(session);
        }

        this.IsDirty = false;

        // Evaluate the parent (input) node
        var x = this.Parents[0].Evaluate(session);
        if (x == null)
        {
            throw new InvalidOperationException("Input to Convolution operation is null.");
        }

        // Dynamically infer the filter shape and validate
        var inputShape = x.Shape;
        if (inputShape.Dimensions[2] != this._filter.Result.Shape.Dimensions[2])
        {
            throw new ArgumentException("Input depth does not match filter depth.");
        }

        // Allocate result volume if needed
        if (this.Result == null || !Equals(this._lastInputShape, inputShape))
        {
            this._lastInputShape = new Shape(inputShape);

            var outputDepth = this.FilterCount;
            var outputWidth =
                (int)Math.Floor((inputShape.Dimensions[0] + this.Pad * 2 - this.Width) / (double)this.Stride + 1);
            var outputHeight =
                (int)Math.Floor((inputShape.Dimensions[1] + this.Pad * 2 - this.Height) / (double)this.Stride + 1);

            this.Result?.Dispose();
            this.Result =
                Volume<T>.CreateEmpty(new Shape(outputWidth, outputHeight, outputDepth, inputShape.Dimensions[3]));
        }

        // Perform convolution
        x.Convolution(this._filter.Evaluate(session), this.Pad, this.Pad, this.Stride, this.Result);

        return base.Evaluate(session);
    }


    public void EvaluateGradient(Session<T> session)
    {
        if (this._lastGradientComputeStep == session.Step)
        {
            return;
        }

        this._lastGradientComputeStep = session.Step;

        var filter = this._filter.Evaluate(session);

        if (this.FilterGradient == null || !Equals(filter.Shape, this.FilterGradient.Shape))
        {
            this.FilterGradient?.Dispose();
            this.FilterGradient = Volume<T>.CreateEmpty(filter.Shape);
        }

        var x = this.Parents[0].Evaluate(session);

        if (this.InputGradient == null || !Equals(x.Shape, this.InputGradient.Shape))
        {
            this.InputGradient?.Dispose();
            this.InputGradient = Volume<T>.CreateEmpty(x.Shape);
        }

        this.FilterGradient.Clear();
        this.InputGradient.Clear();

        var outputGradients = this.Derivative.Evaluate(session);
        x.ConvolutionGradient(filter, outputGradients, this.FilterGradient, this.Pad, this.Stride, this.InputGradient);
    }

    public override Dictionary<string, object> GetData()
    {
        var data = base.GetData();

        data["Stride"] = this.Stride;
        data["Pad"] = this.Pad;
        data["FilterCount"] = this.FilterCount;
        data["Width"] = this.Width;
        data["Height"] = this.Height;
        data["Filter"] = this._filter; // Serialize filter for deserialization

        return data;
    }

    public override string ToString()
    {
        return $"Conv({this.Parents[0]})";
    }
}