using System.Numerics;
using ConvNetSharp.V2.Ops;

namespace ConvNetSharp.V2.Graph;

/// <summary>
///     Walk the computation graph and execute func on each node
/// </summary>
/// <typeparam name="T"></typeparam>
public class OpVisitor<T>(Action<Op<T>> func, bool bothWays = false) : IOpVisitor<T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    private readonly Action<Op<T>> func = func ?? throw new ArgumentNullException(nameof(func));
    private readonly HashSet<Op<T>> visited = [];

    public void Visit(Op<T> op)
    {
        if (op == null)
        {
            return;
        }

        if (this.visited.Contains(op))
        {
            return;
        }

        this.func(op);

        this.visited.Add(op);

        foreach (var parent in op.Parents)
        {
            parent.Accept(this);
        }

        if (bothWays)
        {
            foreach (var child in op.Children)
            {
                child.Accept(this);
            }
        }
    }
}