using System;
using System.Collections.Generic;
using ConvNetSharp.Flow.Graph;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Walk the computation graph and execute func on each node
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class OpVisitor<T> : IOpVisitor<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly bool _bothWays;
        private readonly Action<Op<T>> _func;
        private readonly HashSet<Op<T>> visited = new HashSet<Op<T>>();

        public OpVisitor(Action<Op<T>> func, bool bothWays = false)
        {
            this._bothWays = bothWays;
            this._func = func;
        }

        public void Visit(Op<T> op)
        {
            if (this.visited.Contains(op))
            {
                return;
            }

            this._func(op);

            this.visited.Add(op);

            foreach (var parents in op.Parents)
            {
                parents.Accept(this);
            }

            if (this._bothWays)
            {
                foreach (var child in op.Children)
                {
                    child.Accept(this);
                }
            }
        }
    }
}