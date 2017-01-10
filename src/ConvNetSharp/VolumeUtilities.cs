using System;

namespace ConvNetSharp
{
    public static class VolumeUtilities
    {
        private static readonly Random Random = new Random(RandomUtilities.Seed);

        public enum FlipMode
        {
            LeftRight,
            UpDown,
            Both
        }

        public static Volume Flip(this Volume volume, FlipMode mode)
        {
            Volume result = volume;
            if (mode == FlipMode.LeftRight || mode == FlipMode.Both)
            {
                // flip volume horziontally
                var w = volume.CloneAndZero();
                for (var x = 0; x < volume.Width; x++)
                {
                    for (var y = 0; y < volume.Height; y++)
                    {
                        for (var depth = 0; depth < volume.Depth; depth++)
                        {
                            w.Set(x, y, depth, volume.Get(volume.Width - x - 1, y, depth)); // copy data over
                        }
                    }
                }
                result = w; //swap
            }

            if (mode == FlipMode.UpDown || mode == FlipMode.Both)
            {
                // flip volume horziontally
                var w = volume.CloneAndZero();
                for (var x = 0; x < volume.Width; x++)
                {
                    for (var y = 0; y < volume.Height; y++)
                    {
                        for (var depth = 0; depth < volume.Depth; depth++)
                        {
                            w.Set(x, y, depth, result.Get(x, volume.Height - y - 1, depth)); // copy data over
                        }
                    }
                }
                result = w; //swap
            }

            return result;
        }

        /// <summary>
        ///     Intended for use with data augmentation
        /// </summary>
        /// <param name="volume">Input volume</param>
        /// <param name="crop">Size of output</param>
        /// <param name="dx">Offset wrt incoming volume, of the shift</param>
        /// <param name="dy">Offset wrt incoming volume, of the shift</param>
        /// <param name="flipLeftRight">flip left/right</param>
        /// <returns></returns>
        public static Volume Augment(this Volume volume, int crop, int dx = -1, int dy = -1, bool flipLeftRight = false)
        {
            if (dx == -1)
            {
                dx = Random.Next(volume.Width - crop);
            }

            if (dy == -1)
            {
                dy = Random.Next(volume.Height - crop);
            }

            // randomly sample a crop in the input volume
            Volume w;
            if (crop != volume.Width || dx != 0 || dy != 0)
            {
                w = new Volume(crop, crop, volume.Depth, 0.0);
                for (var x = 0; x < crop; x++)
                {
                    for (var y = 0; y < crop; y++)
                    {
                        if (x + dx < 0 || x + dx >= volume.Width || y + dy < 0 || y + dy >= volume.Width)
                        {
                            continue; // oob
                        }

                        for (var depth = 0; depth < volume.Depth; depth++)
                        {
                            w.Set(x, y, depth, volume.Get(x + dx, y + dy, depth)); // copy data over
                        }
                    }
                }
            }
            else
            {
                w = volume;
            }

            if (flipLeftRight)
            {
                w = w.Flip(FlipMode.LeftRight);
            }

            return w;
        }
    }
}