namespace ConvNetSharp.Core.Training.Double
{
    public class AdamTrainer : AdamTrainer<double>
    {
        public AdamTrainer(INet<double> net) : base(net)
        {
        }
    }
}