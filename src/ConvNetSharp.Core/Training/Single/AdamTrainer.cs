namespace ConvNetSharp.Core.Training.Single
{
    public class AdamTrainer : AdamTrainer<float>
    {
        public AdamTrainer(INet<float> net) : base(net)
        {
        }
    }
}