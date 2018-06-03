using System;

namespace ConvNetSharp.Volume.Double
{
    public class VolumeBuilder : VolumeBuilder<double>
    {
        public override Volume<double> Build(VolumeStorage<double> storage, Shape shape)
        {
            if (storage is NcwhVolumeStorage<double> ncwh)
            {
                return new Volume(ncwh.ReShape(shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<double> From(double[] value, Shape shape)
        {
            shape.GuessUnkownDimension(value.Length);

            if (shape.TotalLength != value.Length)
            {
                throw new ArgumentException($"Array size ({value.Length}) and shape ({shape}) are incompatible");
            }

            return new Volume(new NcwhVolumeStorage<double>(value, shape));
        }

        public override Volume<double> Random(Shape shape, double mu = 0, double std = 1.0)
        {
            //RandomUtilities.RandomDoubleArray(shape.TotalLength, mu, std), shape)
            var vol = new Volume(new NcwhVolumeStorage<double>(shape));

            for (var n = 0; n < shape.Dimensions[3]; n++)
            {
                for (var c = 0; c < shape.Dimensions[2]; c++)
                {
                    for (var y = 0; y < shape.Dimensions[1]; y++)
                    {
                        for (var x = 0; x < shape.Dimensions[0]; x++)
                        {
                            vol.Set(x, y, c, n, RandomUtilities.Randn(mu, std));
                        }
                    }
                }
            }

            return vol;
        }

        public override Volume<double> SameAs(VolumeStorage<double> example, Shape shape)
        {
            if (example is NcwhVolumeStorage<double>)
            {
                return new Volume(new NcwhVolumeStorage<double>(shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<double> SameAs(VolumeStorage<double> example, double value, Shape shape)
        {
            if (example is NcwhVolumeStorage<double>)
            {
                return new Volume(new NcwhVolumeStorage<double>(new double[shape.TotalLength].Populate(value), shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<double> SameAs(Shape shape)
        {
            return new Volume(new NcwhVolumeStorage<double>(shape));
        }
    }
}