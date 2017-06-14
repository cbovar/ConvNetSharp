using System;
using ManagedCuda.CudaDNN;

namespace ConvNetSharp.Volume.GPU
{
    internal static class TensorReduceOpExtension
    {
        public static cudnnReduceTensorOp ToCudnn(this TensorReduceOp op)
        {
            switch (op)
            {
                case TensorReduceOp.Add:
                    return cudnnReduceTensorOp.Add;
                case TensorReduceOp.Mul:
                    return cudnnReduceTensorOp.Mul;
                case TensorReduceOp.Min:
                    return cudnnReduceTensorOp.Min;
                case TensorReduceOp.Max:
                    return cudnnReduceTensorOp.Max;
                case TensorReduceOp.AMax:
                    return cudnnReduceTensorOp.AMax;
                case TensorReduceOp.Avg:
                    return cudnnReduceTensorOp.Avg;
                case TensorReduceOp.Norm1:
                    return cudnnReduceTensorOp.Norm1;
                case TensorReduceOp.Norm2:
                    return cudnnReduceTensorOp.Norm2;
                default:
                    throw new ArgumentOutOfRangeException(nameof(op), op, null);
            }
        }
    }
}