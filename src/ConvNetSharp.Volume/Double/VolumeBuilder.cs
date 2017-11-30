using System;
using System.Diagnostics;

namespace ConvNetSharp.Volume.Double
{
    public class VolumeBuilder : VolumeBuilder<double>
    {
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

        public override Volume<double> From(double[] value, Shape shape)
        {
            return new Volume(new NcwhVolumeStorage<double>(value, shape));
        }

        public override Volume<double> Random(Shape shape, double mu = 0, double std = 1.0)
        {
            //RandomUtilities.RandomDoubleArray(shape.TotalLength, mu, std), shape)
            var vol = new Volume(new NcwhVolumeStorage<double>(shape));

            for (int n = 0; n < shape.GetDimension(3); n++)
            {
                for (int c = 0; c < shape.GetDimension(2); c++)
                {
                    for (int y = 0; y < shape.GetDimension(1); y++)
                    {
                        for (int x = 0; x < shape.GetDimension(0); x++)
                        {
                            vol.Set(x, y, c, n, RandomUtilities.Randn(mu, std));
                        }
                    }
                }
            }

            return vol;
        }

        public override Volume<double> SameAs(Shape shape)
        {
            return new Volume(new NcwhVolumeStorage<double>(shape));
        }

        public override Volume<double> Build(VolumeStorage<double> storage, Shape shape)
        {
            var ncwh = storage as NcwhVolumeStorage<double>;
            if (ncwh != null)
            {
                return new Volume(ncwh.ReShape(shape));
            }

            throw new NotImplementedException();
        }
    }
}