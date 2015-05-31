namespace ConvNetSharp
{
    public interface ILastLayer
    {
        double Backward(double y);

        double Backward(double[] y);
    }
}