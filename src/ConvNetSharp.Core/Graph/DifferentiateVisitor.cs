using System;
using ConvNetSharp.Core.Ops;

namespace ConvNetSharp.Core.Graph
{
    internal class DifferentiateVisitor<T> : IOpVisitor<T> where T : struct, IEquatable<T>, IFormattable
    {
        public void Visit(Op<T> op)
        {
            op.Backward();

            foreach (var parent in op.Parents)
            {
                var diff = new DifferentiateVisitor<T>();
                diff.Visit(parent);
            }
        }
    }
}