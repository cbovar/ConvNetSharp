using System.Collections.Generic;
using System.Numerics;

namespace ConvNetSharp.Flow.Ops;

/// <summary>
///     Dense / Fully connected layer is just a convolution of 'neuronCount' 1x1 filters
/// </summary>
/// <typeparam name="T"></typeparam>
public class Dense<T> : Convolution<T> where T : INumber<T>, IMinMaxValue<T>
{
    public Dense(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph, data)
    {
    }

    public Dense(ConvNetSharp<T> graph, Op<T> x, int neuronCount) : base(graph, x, 1, 1, neuronCount)
    {
    }
}