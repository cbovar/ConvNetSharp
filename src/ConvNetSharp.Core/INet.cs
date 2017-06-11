using System;
using System.Collections.Generic;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core
{
    public interface INet<T> where T : struct, IEquatable<T>, IFormattable
    {
        T Backward(Volume<T> y);

        Volume<T> Forward(Volume<T> input, bool isTraining = false);

        T GetCostLoss(Volume<T> input, Volume<T> y);

        List<ParametersAndGradients<T>> GetParametersAndGradients();

        int[] GetPrediction();
    }
}