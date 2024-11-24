using System.Collections.Generic;
using System.Numerics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers;

public class DropoutLayer<T> : LayerBase<T> where T : INumber<T>, IMinMaxValue<T>
{
    public DropoutLayer(Dictionary<string, object> data) : base(data)
    {
        this.DropProbability = (float)data["DropProbability"];
    }

    public DropoutLayer(float dropProbability)
    {
        this.DropProbability = dropProbability;
    }

    public float DropProbability { get; set; }

    public override void Backward(Volume<T> outputGradient)
    {
        this.OutputActivationGradients = outputGradient;

        this.InputActivationGradients.Clear();

        this.OutputActivation.DropoutGradient(this.InputActivation, this.OutputActivationGradients,
            this.DropProbability, this.InputActivationGradients);
    }

    protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
    {
        input.Dropout(isTraining ? this.DropProbability : 0.0f, this.OutputActivation);
        return this.OutputActivation;
    }

    public override Dictionary<string, object> GetData()
    {
        var dico = base.GetData();
        dico["DropProbability"] = this.DropProbability;
        return dico;
    }

    public override void Init(int inputWidth, int inputHeight, int inputDepth)
    {
        base.Init(inputWidth, inputHeight, inputDepth);

        this.OutputDepth = inputDepth;
        this.OutputWidth = inputWidth;
        this.OutputHeight = inputHeight;
    }
}