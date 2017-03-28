using System;
using System.IO;
using System.Net;

namespace FluentMnistDemo
{
    internal class DataSets
    {
        private const string urlMnist = @"http://yann.lecun.com/exdb/mnist/";
        private const string mnistFolder = @"..\Mnist\";
        private const string trainingLabelFile = "train-labels-idx1-ubyte.gz";
        private const string trainingImageFile = "train-images-idx3-ubyte.gz";
        private const string testingLabelFile = "t10k-labels-idx1-ubyte.gz";
        private const string testingImageFile = "t10k-images-idx3-ubyte.gz";

        public DataSet Train { get; set; }

        public DataSet Validation { get; set; }

        public DataSet Test { get; set; }

        private void DownloadFile(string urlFile, string destFilepath)
        {
            if (!File.Exists(destFilepath))
            {
                try
                {
                    using (var client = new WebClient())
                    {
                        client.DownloadFile(urlFile, destFilepath);
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Failed downloading " + urlFile);
                    Console.WriteLine(e.Message);
                }
            }
        }

        public bool Load(int validationSize = 1000)
        {
            Directory.CreateDirectory(mnistFolder);

            var trainingLabelFilePath = Path.Combine(mnistFolder, trainingLabelFile);
            var trainingImageFilePath = Path.Combine(mnistFolder, trainingImageFile);
            var testingLabelFilePath = Path.Combine(mnistFolder, testingLabelFile);
            var testingImageFilePath = Path.Combine(mnistFolder, testingImageFile);

            // Download Mnist files if needed
            Console.WriteLine("Downloading Mnist training files...");
            DownloadFile(urlMnist + trainingLabelFile, trainingLabelFilePath);
            DownloadFile(urlMnist + trainingImageFile, trainingImageFilePath);
            Console.WriteLine("Downloading Mnist testing files...");
            DownloadFile(urlMnist + testingLabelFile, testingLabelFilePath);
            DownloadFile(urlMnist + testingImageFile, testingImageFilePath);

            // Load data
            Console.WriteLine("Loading the datasets...");
            var train_images = MnistReader.Load(trainingLabelFilePath, trainingImageFilePath);
            var testing_images = MnistReader.Load(testingLabelFilePath, testingImageFilePath);

            var valiation_images = train_images.GetRange(train_images.Count - validationSize, validationSize);
            train_images = train_images.GetRange(0, train_images.Count - validationSize);

            if (train_images.Count == 0 || valiation_images.Count == 0 || testing_images.Count == 0)
            {
                Console.WriteLine("Missing Mnist training/testing files.");
                Console.ReadKey();
                return false;
            }

            this.Train = new DataSet(train_images);
            this.Validation = new DataSet(valiation_images);
            this.Test = new DataSet(testing_images);

            return true;
        }
    }
}