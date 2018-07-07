using System;
using ConvNetSharp.Flow.Ops;

namespace ConvNetSharp.Flow.Graph
{
    internal class DifferentiateVisitor<T> : IOpVisitor<T> where T : struct, IEquatable<T>, IFormattable
    {
        public void Visit(Op<T> op)
        {
            if (op.Derivate != null)
            {
                op.Differentiate();
            }

            foreach (var parent in op.Parents)
            {
                var diff = new DifferentiateVisitor<T>();
                diff.Visit(parent);
            }
        }
    }
}