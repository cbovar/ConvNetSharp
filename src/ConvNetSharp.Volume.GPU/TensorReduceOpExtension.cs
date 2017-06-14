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
                    break;
                case TensorReduceOp.Mul:
                    return cudnnReduceTensorOp.Mul;
                    break;
                case TensorReduceOp.Min:
                    return cudnnReduceTensorOp.Min;
                    break;
                case TensorReduceOp.Max:
                    return cudnnReduceTensorOp.Max;
                    break;
                case TensorReduceOp.AMax:
                    return cudnnReduceTensorOp.AMax;
                    break;
                case TensorReduceOp.Avg:
                    return cudnnReduceTensorOp.Avg;
                    break;
                case TensorReduceOp.Norm1:
                    return cudnnReduceTensorOp.Norm1;
                    break;
                case TensorReduceOp.Norm2:
                    return cudnnReduceTensorOp.Norm2;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(op), op, null);
            }
        }
    }
}