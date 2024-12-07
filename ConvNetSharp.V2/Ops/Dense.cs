using System.Numerics;

namespace ConvNetSharp.V2.Ops;

/// <summary>
///     Dense / Fully connected layer is just a convolution of 'neuronCount' 1x1 filters
/// </summary>
/// <typeparam name="T"></typeparam>
public class Dense<T> : Convolution<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    public Dense(ConvNetSharp<T> graph, Op<T> x, int neuronCount)
        : this(graph, x, Volume<T>.CreateEmpty(InferFilterShape(x, neuronCount)))
    {
    }

    public Dense(ConvNetSharp<T> graph, Op<T> x, Volume<T> filter)
        : base(graph, x, filter, 1, 1, filter.Shape.Dimensions[3])
    {
        if (filter.Shape.Dimensions[0] != 1 || filter.Shape.Dimensions[1] != 1)
        {
            throw new ArgumentException("Dense layer filters must have a shape of 1x1.");
        }
    }

    private static Shape InferFilterShape(Op<T> x, int neuronCount)
    {
        // Infer the shape dynamically
        var inputDepth = x.Result?.Shape.Dimensions[2] ??
                         throw new InvalidOperationException("Input shape must be defined.");
        return new Shape(1, 1, inputDepth, neuronCount);
    }
}