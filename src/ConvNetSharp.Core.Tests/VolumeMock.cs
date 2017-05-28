using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Tests
{
    public class VolumeMock : Volume<double>
    {
        public VolumeMock(double value, Shape shape) : base(new NcwhVolumeStorage<double>(new[] {value}, shape))
        {
        }

        public int DoAddCount { get; set; }

        public int DoMultiplyCount { get; set; }

        public int DoNegateCount { get; set; }

        public override void DoAdd(Volume<double> other, Volume<double> result)
        {
            this.DoAddCount++;
        }

        public override void DoMultiply(Volume<double> other, Volume<double> result)
        {
            this.DoMultiplyCount++;
        }

        public override void DoNegate(Volume<double> result)
        {
            this.DoNegateCount++;
        }
    }
}