using System;
using System.Drawing;

namespace ConvNetSharp.Utils
{
    public static class Image
    {
        public static Bitmap ToBitmap<T>(Volume.Volume<T> vol, int n = 0) where T : struct, IEquatable<T>, IFormattable
        {
            var bmp = new Bitmap(vol.Shape.Dimensions[0], vol.Shape.Dimensions[1]);
            bool isBlackAndWhite = vol.Shape.Dimensions[2] == 1;

            for (int y = 0; y < bmp.Height; y++)
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    Color color;
                    if (isBlackAndWhite)
                    {
                        byte c = ToColorComponent(vol.Get(x, y, 0, n));
                        color = Color.FromArgb(c, c, c);
                    }
                    else
                    {
                        byte r = ToColorComponent(vol.Get(x, y, 0,n));
                        byte g = ToColorComponent(vol.Get(x, y, 1,n));
                        byte b = ToColorComponent(vol.Get(x, y, 2,n));
                        color = Color.FromArgb(r, g, b);
                    }
                    bmp.SetPixel(x, y, color);
                }
            }


            return bmp;
        }

        private static byte ToColorComponent<T>(T val) where T : struct, IEquatable<T>, IFormattable
        {
            var d = Math.Abs((double)Convert.ChangeType(val, typeof(double))) * 255.0;
            d = d > 255.0 ? 255.0 : d;

            return (byte)d;
        }
    }
}