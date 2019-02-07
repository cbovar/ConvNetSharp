using System.Collections.Generic;
using System.Linq;

namespace RnnDemo.Flow.GPU
{
    internal class TextData
    {
        public TextData(string rawData)
        {
            this.RawData = rawData;
            this.Vocabulary = rawData.Select(c => c).Distinct().OrderBy(c => c).ToList();
        }

        public string RawData { get; }

        public List<char> Vocabulary { get; }
    }
}