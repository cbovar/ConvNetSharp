using System;

namespace ConvNetSharp.Volume.GPU.Single
{
    public class VolumeBuilder : VolumeBuilder<float>
    {
        public GpuContext Context { get; set; } = GpuContext.Default;

        public override Volume<float> Build(VolumeStorage<float> storage, Shape shape)
        {
            var gpuStorage = storage as VolumeStorage;
            if (gpuStorage != null)
            {
                return new Volume(new VolumeStorage(gpuStorage, shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<float> Random(Shape shape, double mu = 0, double std = 1.0)
        {
            return new Volume(new VolumeStorage(RandomUtilities.RandomSingleArray(shape.TotalLength, mu, std), shape, this.Context));
        }

        public override Volume<float> SameAs(VolumeStorage<float> example, Shape shape)
        {
            var gpuStorage = example as VolumeStorage;
            if (gpuStorage != null)
            {
                return new Volume(new VolumeStorage(shape, gpuStorage.Context));
            }

            throw new NotImplementedException();
        }

        public override Volume<float> SameAs(VolumeStorage<float> example, float value, Shape shape)
        {
            var gpuStorage = example as VolumeStorage;
            if (gpuStorage != null)
            {
                return new Volume(new VolumeStorage(new float[shape.TotalLength].Populate(value), shape, gpuStorage.Context));
            }

            throw new NotImplementedException();
        }

        public override Volume<float> From(float[] value, Shape shape)
        {
            shape.GuessUnkownDimension(value.Length);

            if (shape.TotalLength != value.Length)
            {
                throw new ArgumentException($"Array size ({value.Length}) and shape ({shape}) are incompatible");
            }

            return new Volume(new VolumeStorage(value, shape, this.Context));
        }

        public override Volume<float> SameAs(Shape shape)
        {
            return new Volume(new VolumeStorage(shape, this.Context));
        }
    }
}