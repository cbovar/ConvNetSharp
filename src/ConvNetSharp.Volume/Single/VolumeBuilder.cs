using System;

namespace ConvNetSharp.Volume.Single
{
    public class VolumeBuilder : VolumeBuilder<float>
    {
        public override Volume<float> Build(VolumeStorage<float> storage, Shape shape)
        {
            if (storage is NcwhVolumeStorage<float> ncwh)
            {
                return new Volume(ncwh.ReShape(shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<float> From(float[] value, Shape shape)
        {
            shape.GuessUnknownDimension(value.Length);

            if (shape.TotalLength != value.Length)
            {
                throw new ArgumentException($"Array size ({value.Length}) and shape ({shape}) are incompatible");
            }

            return new Volume(new NcwhVolumeStorage<float>(value, shape));
        }

        public override Volume<float> Random(Shape shape, double mu = 0, double std = 1.0)
        {
            return new Volume(new NcwhVolumeStorage<float>(RandomUtilities.RandomSingleArray(shape.TotalLength, mu, std), shape));
        }

        public override Volume<float> SameAs(VolumeStorage<float> example, Shape shape)
        {
            if (example is NcwhVolumeStorage<float>)
            {
                return new Volume(new NcwhVolumeStorage<float>(shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<float> SameAs(VolumeStorage<float> example, float value, Shape shape)
        {
            if (example is NcwhVolumeStorage<float>)
            {
                return new Volume(new NcwhVolumeStorage<float>(new float[shape.TotalLength].Populate(value), shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<float> SameAs(Shape shape)
        {
            return new Volume(new NcwhVolumeStorage<float>(shape));
        }
    }
}