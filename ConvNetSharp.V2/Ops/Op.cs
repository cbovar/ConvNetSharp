using System.Numerics;
using ConvNetSharp.V2.Graph;

namespace ConvNetSharp.V2.Ops;

public abstract class Op<T> : IDisposable
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    public ConvNetSharp<T> Graph;

    protected Op(ConvNetSharp<T> cns)
    {
        this.Graph = cns;
        Count++;
    }

    public static int Count { get; set; } = 1;

    public bool IsDirty { get; set; } = true;

    public Volume<T>? Result { get; set; }

    public Op<T> Derivative { get; set; }

    public List<Op<T>> Parents { get; } = [];

    public List<Op<T>> Children { get; } = [];

    public virtual string Representation { get; } = "";

    public void Dispose()
    {
        this.Dispose(true);
        GC.SuppressFinalize(this);
    }

    public void Accept(IOpVisitor<T> visitor)
    {
        visitor.Visit(this);
    }

    public void AddParent(Op<T> parent)
    {
        this.Parents.Add(parent);

        if (!this.Children.Contains(this))
        {
            parent.Children.Add(this);
        }
    }

    public abstract void Differentiate();

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Derivative?.Dispose();
        }
    }

    public static void DisposeGraph(Op<T> root)
    {
        var visitor = new OpVisitor<T>(op =>
        {
            if (!(op is INamedOp<T>)) // not sure about this
            {
                op.Dispose();
            }
        }, true);
        root?.Accept(visitor);
    }

    public virtual Volume<T> Evaluate(Session<T> session)
    {
        this.Evaluated?.Invoke(this, EventArgs.Empty);
        return this.Result;
    }

    public event EventHandler Evaluated;

    public virtual Dictionary<string, object> GetData()
    {
        return new Dictionary<string, object>();
    }

    public static Op<T> operator +(Op<T> left, Op<T> right)
    {
        if (left.Graph != right.Graph)
        {
            throw new InvalidOperationException("Operations belong to different graphs and cannot be combined.");
        }

        return new Add<T>(left.Graph, left, right);
    }

    public static Op<T> operator /(Op<T> left, Op<T> right)
    {
        if (left.Graph != right.Graph)
        {
            throw new InvalidOperationException("Operations belong to different graphs and cannot be combined.");
        }

        return new Div<T>(left.Graph, left, right);
    }

    public static Op<T> operator *(Op<T> left, Op<T> right)
    {
        if (left.Graph != right.Graph)
        {
            throw new InvalidOperationException("Operations belong to different graphs and cannot be combined.");
        }

        return new Mult<T>(left.Graph, left, right);
    }

    public static Op<T> operator ^(Op<T> left, Op<T> right)
    {
        if (left.Graph != right.Graph)
        {
            throw new InvalidOperationException("Operations belong to different graphs and cannot be combined.");
        }

        return new Power<T>(left.Graph, left, right);
    }

    public static Op<T> operator -(Op<T> left, Op<T> right)
    {
        if (left.Graph != right.Graph)
        {
            throw new InvalidOperationException("Operations belong to different graphs and cannot be combined.");
        }

        return new Add<T>(left.Graph, left, -right);
    }

    public static Op<T> operator -(Op<T> x)
    {
        return new Negate<T>(x.Graph, x);
    }

    public static implicit operator Op<T>(T x)
    {
        return new Const<T>(ConvNetSharp<T>.Current, x, x.ToString()); // Use Default graph => can we do better ?
    }

    public void RegisterDerivative(Op<T> d)
    {
        if (this.Derivative == null)
        {
            this.Derivative = d;
        }
        else
        {
            this.Derivative += d;
        }
    }

    public void RemoveParent(Op<T> parent)
    {
        this.Parents.Remove(parent);
        parent.Children.Remove(this);
    }

    public void SetDirty()
    {
        this.IsDirty = true;
        foreach (var child in this.Children)
        {
            if (!child.IsDirty)
            {
                child.SetDirty();
            }
        }
    }
}