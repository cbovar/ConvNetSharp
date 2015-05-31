namespace ConvNetSharp
{

    public enum Activation
    {
        Undefined,
        Relu,
        Sigmoid,
        Tanh,
        Maxout
    }

    public interface IConvLayer
    {
        double BiasPref { get; set; }

        Activation Activation { get; }

        int GroupSize { get; }
    }
}