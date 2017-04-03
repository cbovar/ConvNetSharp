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

        public override Volume<double> SameAs(double[] value, Shape shape)
        {
            return new Volume(new NcwhVolumeStorage<double>(value, shape));
        }

        public override Volume<double> Random(Shape shape, double mu = 0, double std = 1.0)
        {
            return new Volume(new NcwhVolumeStorage<double>(RandomUtilities.RandomDoubleArray(shape.TotalLength, mu, std), shape));
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