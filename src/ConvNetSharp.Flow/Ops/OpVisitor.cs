using System;
using System.Collections.Generic;
using System.Numerics;
using ConvNetSharp.Flow.Graph;

namespace ConvNetSharp.Flow.Ops;

/// <summary>
///     Walk the computation graph and execute func on each node
/// </summary>
/// <typeparam name="T"></typeparam>
public class OpVisitor<T>(Action<Op<T>> func, bool bothWays = false) : IOpVisitor<T>
    where T : INumber<T>, IMinMaxValue<T>
{
    private readonly HashSet<Op<T>> visited = [];

    public void Visit(Op<T> op)
    {
        if (this.visited.Contains(op))
        {
            return;
        }

        func(op);

        this.visited.Add(op);

        foreach (var parents in op.Parents)
        {
            parents.Accept(this);
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