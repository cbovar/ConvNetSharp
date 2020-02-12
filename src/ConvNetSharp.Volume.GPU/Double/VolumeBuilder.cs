using System;

namespace ConvNetSharp.Volume.GPU.Double
{
    public class VolumeBuilder : VolumeBuilder<double>
    {
        public GpuContext Context { get; set; } = GpuContext.Default;

        public override Volume<double> Build(VolumeStorage<double> storage, Shape shape)
        {
            if (storage is VolumeStorage gpuStorage)
            {
                return new Volume(new VolumeStorage(gpuStorage, shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<double> Random(Shape shape, double mu = 0, double std = 1.0)
        {
            return new Volume(new VolumeStorage(RandomUtilities.RandomDoubleArray(shape.TotalLength, mu, std), shape, this.Context));
        }

        public override Volume<double> SameAs(VolumeStorage<double> example, Shape shape)
        {
            if (example is VolumeStorage gpuStorage)
            {
                return new Volume(new VolumeStorage(shape, gpuStorage.Context));
            }

            throw new NotImplementedException();
        }

        public override Volume<double> SameAs(VolumeStorage<double> example, double value, Shape shape)
        {
            if (example is VolumeStorage gpuStorage)
            {
                return new Volume(new VolumeStorage(new double[shape.TotalLength].Populate(value), shape, gpuStorage.Context));
            }

            throw new NotImplementedException();
        }

        public override Volume<double> From(double[] value, Shape shape)
        {
            shape.GuessUnknownDimension(value.Length);

            if (shape.TotalLength != value.Length)
            {
                throw new ArgumentException($"Array size ({value.Length}) and shape ({shape}) are incompatible");
            }

            return new Volume(new VolumeStorage(value, shape, this.Context));
        }

        public override Volume<double> SameAs(Shape shape)
        {
            return new Volume(new VolumeStorage(shape, this.Context));
        }
    }
}