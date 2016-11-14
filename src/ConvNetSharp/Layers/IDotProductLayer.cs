namespace ConvNetSharp.Layers
{
    public interface IDotProductLayer
    {
        double BiasPref { get; set; }

        Activation Activation { get; }

        int GroupSize { get; }
    }
}