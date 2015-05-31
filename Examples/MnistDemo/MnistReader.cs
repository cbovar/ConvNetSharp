using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MnistDemo
{
    public static class MnistReader
    {
        public static List<MnistEntry> Load(string labelFile, string imageFile, int maxItem = -1)
        {
            List<int> label = LoadLabels(labelFile, maxItem);
            List<byte[]> images = LoadImages(imageFile, maxItem);

            return label.Select((t, i) => new MnistEntry {Label = t, Image = images[i]}).ToList();
        }

        private static int ReverseBytes(int v)
        {
            byte[] intAsBytes = BitConverter.GetBytes(v);
            Array.Reverse(intAsBytes);
            return BitConverter.ToInt32(intAsBytes, 0);
        }

        private static List<int> LoadLabels(string filename, int maxItem = -1)
        {
            var result = new List<int>();

            using (var reader = new BinaryReader((File.Open(filename, FileMode.Open))))
            {
                var magicNumber = ReverseBytes(reader.ReadInt32());
                var numberOfItem = ReverseBytes(reader.ReadInt32());
                if (maxItem != -1)
                {
                    numberOfItem = Math.Min(numberOfItem, maxItem);
                }

                for (var i = 0; i < numberOfItem; i++)
                {
                    result.Add(reader.ReadByte());
                }
            }

            return result;
        }

        private static List<byte[]> LoadImages(string filename, int maxItem = -1)
        {
            var result = new List<byte[]>();

            using (var reader = new BinaryReader((File.Open(filename, FileMode.Open))))
            {
                var magicNumber = ReverseBytes(reader.ReadInt32());
                var numberOfImage = ReverseBytes(reader.ReadInt32());
                var rowCount = ReverseBytes(reader.ReadInt32());
                var columnCount = ReverseBytes(reader.ReadInt32());
                if (maxItem != -1)
                {
                    numberOfImage = Math.Min(numberOfImage, maxItem);
                }

                for (var i = 0; i < numberOfImage; i++)
                {
                    byte[] image = reader.ReadBytes(rowCount*columnCount);
                    result.Add(image);
                }
            }

            return result;
        }
    }
}