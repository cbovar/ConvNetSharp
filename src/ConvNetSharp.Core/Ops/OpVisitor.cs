using System;
using ConvNetSharp.Core.Graph;

namespace ConvNetSharp.Core.Ops
{
    /// <summary>
    /// Walk the computation graph and execute func on each node
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class OpVisitor<T> : IOpVisitor<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Action<Op<T>> _func;

        public OpVisitor(Action<Op<T>> func)
        {
            this._func = func;
        }

        public void Visit(Op<T> op)
        {
            this._func(op);

            foreach (var parents in op.Parents)
            {
                parents.Accept(this);
            }
        }
    }
}