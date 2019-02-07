using System;
using System.IO;

namespace RnnDemo.Flow.GPU
{
    internal class Program
    {
        /// <summary>
        /// This demo is has some issues:
        /// - Network never manages to get every word right (not matter the size of hidden state)
        /// - When starting a new epoch, loss increases. This is probably due to discontinuity in the state.
        /// - Text generation is not impressive at all. Some real words are sometimes generated but it's mostly garbage.
        /// 
        /// TODO: use LSTM
        /// </summary>
        private static void Main()
        {
            // Load Shakespeare data
            //var book = new StringBuilder();
            //foreach (var file in Directory.EnumerateFiles(@"./shakespeare/", "*.txt"))
            //{
            //    using (var sr = new StreamReader(new FileStream(file, FileMode.Open, FileAccess.Read)))
            //    {
            //        book.Append(sr.ReadToEnd());
            //    }
            //}

            //var textData = new TextData(book.ToString());
            var textData = new TextData(File.ReadAllText("Simple.txt"));

            // Start learning
            var rnnDemo = new RnnDemo(textData, 3, 100, 300, 0.01f);

            while (true)
            {
                Console.Clear();
                Console.WriteLine(Environment.NewLine + "---- Rnn demo ----");
                Console.WriteLine("0) Learning");
                Console.WriteLine("1) Text generation");
                Console.WriteLine("2) Exit");

                var c = Console.ReadKey();
                switch (c.KeyChar)
                {
                    case '2':
                        return;
                    case '0':
                        rnnDemo.Learn();
                        break;
                    case '1':
                        rnnDemo.GenerateText();
                        break;
                }
            }
        }
    }
}