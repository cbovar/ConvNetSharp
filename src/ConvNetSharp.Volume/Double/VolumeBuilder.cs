using System;

namespace ConvNetSharp.Volume.Double
{
    public class VolumeBuilder : VolumeBuilder<double>
    {
        public override Volume<double> Build(VolumeStorage<double> storage, Shape shape)
        {
            var ncwh = storage as NcwhVolumeStorage<double>;
            if (ncwh != null)
            {
                return new Volume(ncwh.ReShape(shape));
            }

            throw new NotImplementedException();
        }

        public override Volume<double> Random(Shape shape, double mu = 0, double std = 1.0)
        {
            var vol = new Volume(new NcwhVolumeStorage<double>(shape));

            for (var n = 0; n < shape.GetDimension(3); n++)
            {
                for (var c = 0; c < shape.GetDimension(2); c++)
                {
                    for (var y = 0; y < shape.GetDimension(1); y++)
                    {
                        for (var x = 0; x < shape.GetDimension(0); x++)
                        {
                            vol.Set(new[] { x, y, c, n }, RandomUtilities.Randn(mu, std));
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

        public override Volume<double> SameAs(double[] value, Shape shape)
        {
            return new Volume(new NcwhVolumeStorage<double>(value, shape));
        }

        public override Volume<double> SameAs(Shape shape)
        {
            return new Volume(new NcwhVolumeStorage<double>(shape));
        }
    }
}