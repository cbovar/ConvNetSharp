namespace ConvNetSharp.Layers
{
    public class ParametersAndGradients
    {
        public double[] Parameters { get; set; }

        public double[] Gradients { get; set; }

        public double? L2DecayMul { get; set; }

        public double? L1DecayMul { get; set; }
    }
}