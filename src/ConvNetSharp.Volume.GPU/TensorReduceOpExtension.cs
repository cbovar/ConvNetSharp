using System;
using ManagedCuda.CudaDNN;

namespace ConvNetSharp.Volume.GPU;

internal static class TensorReduceOpExtension
{
    public static cudnnReduceTensorOp ToCudnn(this TensorReduceOp op)
    {
        return op switch
        {
            TensorReduceOp.Add => cudnnReduceTensorOp.Add,
            TensorReduceOp.Mul => cudnnReduceTensorOp.Mul,
            TensorReduceOp.Min => cudnnReduceTensorOp.Min,
            TensorReduceOp.Max => cudnnReduceTensorOp.Max,
            TensorReduceOp.AMax => cudnnReduceTensorOp.AMax,
            TensorReduceOp.Avg => cudnnReduceTensorOp.Avg,
            TensorReduceOp.Norm1 => cudnnReduceTensorOp.Norm1,
            TensorReduceOp.Norm2 => cudnnReduceTensorOp.Norm2,
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, null)
        };
    }
}