using System.Collections.Generic;
using ConvNetSharp.Layers;

namespace ConvNetSharp
{
    public interface INet
    {
        double Backward(double[] y);

        double Backward(double y);

        IVolume Forward(bool isTraining = false, params IVolume[] inputs);

        double GetCostLoss(IVolume volume, double[] y);

        double GetCostLoss(IVolume volume, double y);

        List<ParametersAndGradients> GetParametersAndGradients();

        int GetPrediction();
    }
}