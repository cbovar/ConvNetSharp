using System;

namespace ConvNetSharp.Volume.Single
{
    public class Volume : Volume<float>
    {
        public Volume(float[] array, Shape shape) : this(new NcwhVolumeStorage<float>(array, shape))
        {
        }

        public Volume(VolumeStorage<float> storage) : base(storage)
        {
        }

        public override void DoAdd(Volume<float> other, Volume<float> result)
        {
            this.Storage.MapEx((x, y) => x + y, other.Storage, result.Storage);
        }

        public override void DoNegate(Volume<float> result)
        {
            this.Storage.Map(x => -x, result.Storage);
        }

        public override void DoMultiply(Volume<float> other, Volume<float> result)
        {
            if (this.Shape.Equals(other.Shape))
            {
                this.Storage.Map((left, right) => left * right, other.Storage, result.Storage);
            }
            else
            {
                //Todo: broadcast
                throw new NotImplementedException();
            }
        }
    }
}