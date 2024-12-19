using System.Numerics;
using ConvNetSharp.V2.Ops;

namespace ConvNetSharp.V2;

/// <summary>
///     Class containing convenience methods to build a computation graph
/// </summary>
/// <typeparam name="T"></typeparam>
public class ConvNetSharp<T> : IScopeManager where T : struct, INumber<T>, IMinMaxValue<T>
{
    public static readonly T Zero = T.Zero;
    public static readonly T One = T.One;
    private static readonly AsyncLocal<ConvNetSharp<T>> _current = new();
    private readonly Stack<string> _scopes = new();

    // Current Active Instance
    public static ConvNetSharp<T> Current
    {
        get
        {
            if (_current.Value == null)
            {
                // Automatically activate a new default instance
                var defaultInstance = new ConvNetSharp<T>();
                SetActive(defaultInstance);
            }

            return _current.Value;
        }
        set => _current.Value = value;
    }


    // Scoped Context
    public void RegisterScope(string name)
    {
        if (this._scopes.Count > 0 && this._scopes.Peek() == name)
        {
            throw new InvalidOperationException($"Scope '{name}' is already registered.");
        }

        this._scopes.Push(name);
    }

    public void ReleaseScope(string name)
    {
        if (this._scopes.Count == 0)
        {
            throw new InvalidOperationException("Cannot release scope; the stack is empty.");
        }

        var currentScope = this._scopes.Pop();
        if (currentScope != name)
        {
            throw new InvalidOperationException(
                $"Attempted to release scope '{name}', but the current scope is '{currentScope}'.");
        }
    }

    // Set Active Instance
    public static IDisposable SetActive(ConvNetSharp<T> instance)
    {
        var previous = _current.Value;
        Current = instance;

        return new DisposableScope(() => Current = previous);
    }

    public Op<T> Assign(Op<T> valueOp, Op<T> op)
    {
        return new Assign<T>(this, valueOp, op);
    }

    public Op<T> Concat(Op<T> x, Op<T> y)
    {
        return new Concat<T>(this, x, y);
    }

    public Const<T> Const(Volume<T> v, string name)
    {
        return new Const<T>(this, v, name);
    }

    public Const<T> Const(T x, string name)
    {
        return new Const<T>(this, x, name);
    }

    public Convolution<T> Conv(Op<T> x, Volume<T> filter, int width, int height, int filterCount, int stride = 1,
        int pad = 0)
    {
        if (filter == null)
        {
            throw new ArgumentNullException(nameof(filter), "Filter must be explicitly provided.");
        }

        return new Convolution<T>(this, x, filter, width, height, filterCount, stride, pad);
    }

    public Op<T> CrossEntropyLoss(Op<T> x, Op<T> y)
    {
        return new SoftmaxCrossEntropy<T>(this, x, y);
    }

    public Dense<T> Dense(Op<T> x, int neuronCount)
    {
        return new Dense<T>(this, x, neuronCount);
    }

    public Op<T> Dropout(Op<T> x, Op<T> dropoutProbability)
    {
        return new Dropout<T>(this, x, dropoutProbability);
    }

    public Op<T> Exp(Op<T> x)
    {
        return new Exp<T>(this, x);
    }

    public Op<T> Extract(Op<T> x, Op<T> length, Op<T> offset)
    {
        return new Extract<T>(this, x, length, offset);
    }

    public Op<T> Flatten(Op<T> x)
    {
        return this.Reshape(x, new Shape(1, 1, -1, V2.Shape.Keep));
    }

    public Op<T> LeakyRelu(Op<T> x, T alpha)
    {
        return new LeakyRelu<T>(this, x, alpha);
    }

    public Op<T> LeakyReluGradient(Op<T> y, Op<T> derivative, T alpha)
    {
        return new LeakyReluGradient<T>(this, y, derivative, alpha);
    }

    public Op<T> Log(Op<T> x)
    {
        return new Log<T>(this, x);
    }

    public Op<T> MatMult(Op<T> x, Op<T> y)
    {
        return new MatMult<T>(this, x, y);
    }

    public Op<T> Max(Op<T> x)
    {
        return new Max<T>(this, x);
    }

    public Op<T> Negate(Op<T> x)
    {
        return new Negate<T>(this, x);
    }

    public PlaceHolder<T> PlaceHolder(string name)
    {
        return new PlaceHolder<T>(this, name);
    }

    public Pool<T> Pool(Op<T> x, int width, int height, int horizontalPad, int verticalPad, int horizontalStride,
        int verticalStride)
    {
        return new Pool<T>(this, x, width, height, horizontalPad, verticalPad, horizontalStride, verticalStride);
    }

    public Op<T> Power(Op<T> u, Op<T> v)
    {
        return new Power<T>(this, u, v);
    }

    public Op<T> Relu(Op<T> x)
    {
        return new Activation<T>(this, x, ActivationType.Relu);
    }

    public Op<T> Reshape(Op<T> x, Shape shape)
    {
        return new Reshape<T>(this, x, shape);
    }

    public Op<T> Reshape(Op<T> x, Op<T> shape)
    {
        return new Reshape<T>(this, x, shape);
    }

    public ScopedContext<T> Scope(string name)
    {
        return new ScopedContext<T>(name, this);
    }

    public Op<T> Shape(Op<T> x)
    {
        return new Shape<T>(this, x);
    }

    public Op<T> Sigmoid(Op<T> x)
    {
        return new Activation<T>(this, x, ActivationType.Sigmoid);
    }

    public Op<T> Softmax(Op<T> x)
    {
        return new Softmax<T>(this, x);
    }

    public Op<T> Sqrt(Op<T> x)
    {
        return new Sqrt<T>(this, x);
    }

    public Op<T> Sum(Op<T> x, Shape shape)
    {
        return new Sum<T>(this, x, shape);
    }

    public Op<T> Sum(Op<T> x, Op<T> shape)
    {
        return new Sum<T>(this, x, shape);
    }

    public Op<T> Tanh(Op<T> x)
    {
        return new Activation<T>(this, x, ActivationType.Tanh);
    }

    public Op<T> Tile(Op<T> x, Op<T> reps)
    {
        return new Tile<T>(this, x, reps);
    }

    public Op<T> Transpose(Op<T> x)
    {
        return new Transpose<T>(this, x);
    }

    public Variable<T> Variable(Volume<T> v, string name, bool isLearnable = false)
    {
        return new Variable<T>(this, v, this.GetScopedName(name), isLearnable);
    }

    public Variable<T> Variable(Shape shape, string name, bool isLearnable = false)
    {
        return this.Variable(Volume<T>.CreateEmpty(shape), name, isLearnable);
    }

    private string GetScopedName(string name)
    {
        var scopedPath = string.Join("/", this._scopes.Reverse());
        if (!string.IsNullOrEmpty(scopedPath) && !name.StartsWith(scopedPath))
        {
            return $"{scopedPath}/{name}";
        }

        return name;
    }

    public Variable<T> Variable(string name, bool isLearnable = false)
    {
        return new Variable<T>(this, null, this.GetScopedName(name), isLearnable);
    }

    private class DisposableScope(Action onDispose) : IDisposable
    {
        public void Dispose()
        {
            onDispose();
        }
    }
}