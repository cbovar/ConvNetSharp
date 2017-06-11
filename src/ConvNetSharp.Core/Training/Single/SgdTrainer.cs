namespace ConvNetSharp.Core.Training.Single
{
    public class SgdTrainer : SgdTrainer<float>
    {
        public SgdTrainer(INet<float> net) : base(net)
        {
        }
    }
}