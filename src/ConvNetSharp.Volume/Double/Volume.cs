using System;

namespace ConvNetSharp.Volume.Double
{
    public class Volume : Volume<double>
    {
        public Volume(double[] array, Shape shape) : this(new NcwhVolumeStorage<double>(array, shape))
        {
        }

        public Volume(VolumeStorage<double> storage) : base(storage)
        {
        }

        public override void DoAdd(Volume<double> other, Volume<double> result)
        {
            this.Storage.MapEx((x, y) => x + y, other.Storage, result.Storage);
        }

        public override void DoNegate(Volume<double> result)
        {
            this.Storage.Map(x => -x, result.Storage);
        }

        public override void DoMultiply(Volume<double> other, Volume<double> result)
        {
            if (this.Shape.Equals(other.Shape))
            {
                this.Storage.Map((left, right) => left * right, other.Storage, result.Storage);
            }
            else
            {
                //ToDO: broadcast
                throw new NotImplementedException();
            }
        }
    }
}