using System.Numerics;
using ConvNetSharp.V2.Ops;

namespace ConvNetSharp.V2.Graph;

internal class DifferentiateVisitor<T> : IOpVisitor<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public void Visit(Op<T> op)
    {
        if (op.Derivative != null)
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