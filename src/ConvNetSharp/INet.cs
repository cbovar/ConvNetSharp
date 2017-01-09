using System.Collections.Generic;
using ConvNetSharp.Layers;

namespace ConvNetSharp
{
    public interface INet
    {
        double Backward(double[] y);

        double Backward(double y);

        IVolume Forward(IVolume input, bool isTraining = false);

        IVolume Forward(IVolume[] inputs, bool isTraining = false);

        double GetCostLoss(IVolume input, double[] y);

        double GetCostLoss(IVolume input, double y);

        List<ParametersAndGradients> GetParametersAndGradients();

        int GetPrediction();
    }
}