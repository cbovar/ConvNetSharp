using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Flow.Serialization;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.GPU.Single;

namespace RnnDemo.Flow
{
    internal class RnnDemo
    {
        private readonly int _batchSize;
        private readonly TextData _data;
        private readonly float _learningRate;
        private readonly int _numClasses; // Vocabulary size
        private readonly int _numSteps; // Size of text to present to network

        private readonly Random _random = new Random();
        private readonly int _stateSize; // Rnn cell internal state size

        private Variable<float> _bh;
        private Variable<float> _by;

        private ConvNetSharp<float> _cns;

        private PlaceHolder<float> _dropoutProba;

        private Op<float> _finalState;
        private Op<float> _output;
        private PlaceHolder<float> _temperature;
        private PlaceHolder<float> _trainingState;
        private Variable<float> _whh;
        private Variable<float> _wxh;
        private Variable<float> _wyh;
        private PlaceHolder<float> _y;

        public RnnDemo(TextData data, int numSteps, int batchSize, int stateSize, float learningRate)
        {
            BuilderInstance<float>.Volume = new VolumeBuilder(); // Use GPU volume builder

            this._data = data;
            this._numSteps = numSteps;
            this._batchSize = batchSize;
            this._numClasses = data.Vocabulary.Count;
            this._stateSize = stateSize;
            this._learningRate = learningRate;
        }

        private IEnumerable<Tuple<List<Volume<float>>, Volume<float>>> GenerateBatch()
        {
            var rnnOutput = BuilderInstance<float>.Volume.SameAs(new Shape(1, 1, this._data.Vocabulary.Count, this._batchSize));
            var rnnInput = new List<Volume<float>>();
            for (var i = 0; i < this._numSteps; i++)
            {
                rnnInput.Add(BuilderInstance<float>.Volume.SameAs(new Shape(this._data.Vocabulary.Count, 1, 1, this._batchSize)));
            }

            var queue = new Queue<char>();
            for (var k = 0; k < this._data.RawData.Length - 1; k++)
            {
                var c = this._data.RawData[k];
                queue.Enqueue(c);
                if (queue.Count > this._numSteps + this._batchSize)
                {
                    queue.Dequeue();
                }

                if (queue.Count == this._numSteps + this._batchSize)
                {
                    var current = queue.ToList();

                    rnnOutput.Clear();
                    for (var i = 0; i < this._numSteps; i++)
                    {
                        rnnInput[i].Clear();
                    }

                    for (var n = 0; n < this._batchSize; n++)
                    {
                        // output
                        var indexY = this._data.Vocabulary.IndexOf(current[this._numSteps + n]); // get next character => it's what we are trying to guess
                        rnnOutput.Set(0, 0, indexY, n, 1.0f);

                        // input
                        for (var j = 0; j < this._numSteps; j++)
                        {
                            var index = this._data.Vocabulary.IndexOf(current[n + j]);
                            rnnInput[j].Set(index, 0, 0, n, 1.0f);
                        }
                    }

                    yield return new Tuple<List<Volume<float>>, Volume<float>>(rnnInput, rnnOutput);
                }
            }
        }

        public void GenerateText()
        {
            this._output = SerializationExtensions.Load<float>("MyNetwork", false)[0];

            var initState = BuilderInstance<float>.Volume.SameAs(new Shape(this._stateSize, 1, 1, 1));
            Volume<float> temperature = 1.0f; // increase this for more creativity (and more spelling mistakes)
            Volume<float> dropoutProba = 0.0f;

            var input = new List<Volume<float>>();

            initState.Clear();

            // Seed
            var inputchar = new char[this._numSteps];
            inputchar[0] = 'n';
            inputchar[1] = 'o';

            for (var i = 0; i < this._numSteps; i++)
            {
                Console.Write(inputchar[i]);
                input.Add(BuilderInstance<float>.Volume.SameAs(new Shape(this._data.Vocabulary.Count, 1, 1, 1)));
            }

            do
            {
                using (var session = new Session<float>())
                {
                    for (var i = 0; i < this._numSteps; i++)
                    {
                        input[i].Clear();
                        input[i].Set(this._data.Vocabulary.IndexOf(inputchar[i]), 0, 0, 0, 1);
                    }

                    var dico = new Dictionary<string, Volume<float>>
                    {
                        {"initState", initState},
                        {"temperature", temperature},
                        {"dropoutProba", dropoutProba}
                    };
                    for (var i = 0; i < this._numSteps; i++)
                    {
                        dico["x" + (i + 1)] = input[i];
                    }

                    var result = session.Run(this._output, dico);
                    var c3 = ToChar(3, result);
                    Console.Write(c3);

                    for (var i = 1; i < this._numSteps; i++)
                    {
                        inputchar[i - 1] = inputchar[i];
                    }

                    inputchar[this._numSteps - 1] = c3;

                    initState = session.GetVariableByName(this._output, "initState").Result.Clone(); // re inject
                }
            } while (!Console.KeyAvailable);
        }

        /// <summary>
        /// Create a new neural network
        /// </summary>
        public void CreateNetwork()
        {
            this._cns = new ConvNetSharp<float>();

            this._whh = this._cns.Variable(BuilderInstance<float>.Volume.Random(new Shape(this._stateSize, this._stateSize, 1, 1), 0, 0.01), "Whh", true);
            this._wxh = this._cns.Variable(BuilderInstance<float>.Volume.Random(new Shape(this._stateSize, this._numClasses, 1, 1), 0, 0.01), "Wxh", true);
            this._wyh = this._cns.Variable(BuilderInstance<float>.Volume.Random(new Shape(this._numClasses, this._stateSize, 1, 1), 0, 0.01), "Wyh", true);
            this._bh = this._cns.Variable(BuilderInstance<float>.Volume.SameAs(new Shape(this._stateSize, 1, 1, 1)), "bh", true);
            this._by = this._cns.Variable(BuilderInstance<float>.Volume.SameAs(new Shape(this._numClasses, 1, 1, 1)), "by", true);

            this._dropoutProba = this._cns.PlaceHolder("dropoutProba");
            this._trainingState = this._cns.PlaceHolder("initState");
            this._y = this._cns.PlaceHolder("y"); // Will hold ground truth
            this._temperature = this._cns.PlaceHolder("temperature");

            Op<float> state = this._trainingState;
            Op<float> finalOutput = null;

            // Create a RnnCell for each step. RnnCell accepts an input and the previous layer state
            for (var i = 0; i < this._numSteps; i++)
            {
                var x = this._cns.PlaceHolder("x" + (i + 1)); // create a place holder for each step (on for each input character)
                var result = RnnCell(x, state);
                state = result.Item2;
                finalOutput = this._cns.Dropout(result.Item1, this._dropoutProba);
            }

            this._finalState = state;
            this._output = this._cns.Softmax(this._cns.Reshape(finalOutput, new Shape(1, 1, -1, Shape.Keep)) / this._temperature);
        }

        public void Learn()
        {
            CreateNetwork();

            Run();
        }

        /// <summary>
        ///     Creates a simple Rnn cell
        /// </summary>
        /// <param name="x">Input at this step. Shape should be [Vocabulary size, 1, 1, Batch size]</param>
        /// <param name="state">
        ///     Previous layer state (or initial state for the first step). Shape should be [State Size, 1, 1,
        ///     Batch Size]
        /// </param>
        /// <returns></returns>
        private Tuple<Op<float>, Op<float>> RnnCell(Op<float> x, Op<float> state)
        {
            var hiddenState = this._cns.Tanh(this._cns.MatMult(x, this._wxh) + this._cns.MatMult(state, this._whh) + this._bh);
            var output = this._cns.MatMult(hiddenState, this._wyh) + this._by;

            return new Tuple<Op<float>, Op<float>>(output, hiddenState);
        }

        public void Run()
        {
            var loss = this._cns.CrossEntropyLoss(this._output, this._y);
            var optimizer = new AdamOptimizer<float>(this._cns, this._learningRate, 0.9f, 0.999f, 1e-08f); //new GradientDescentOptimizer<float>(this._cns, this._learningRate);

            var initState = BuilderInstance<float>.Volume.SameAs(new Shape(this._stateSize, 1, 1, this._batchSize));
            Volume<float> temperature = 1.0f;
            Volume<float> dropoutProba = 0.2f;

            if (File.Exists("loss.csv")) File.Delete("loss.csv");

            using (var session = new Session<float>())
            {
                session.Differentiate(loss);
                var count = 0;
                do
                {
                    Console.WriteLine(Environment.NewLine + "NEW EPOCH");

                    foreach (var batch in GenerateBatch())
                    {
                        var dico = new Dictionary<string, Volume<float>>
                        {
                            {"initState", initState},
                            {"temperature", temperature},
                            {"dropoutProba", dropoutProba},
                            {"y", batch.Item2}
                        };

                        for (var i = 0; i < this._numSteps; i++)
                        {
                            dico["x" + (i + 1)] = batch.Item1[i];
                        }

                        var result = session.Run(this._output, dico);
                        var finalState = this._finalState.Result.Clone();
                        initState = finalState; // re inject state

                        // Display batch 0 guess. 'X' means incorrectly guessed
                        var lastOutput = ToChar(1, result);
                        var expected = ToChar(1, batch.Item2);
                        Console.Write(lastOutput == expected ? lastOutput : 'X');

                        // Compute loss
                        var currentCost = session.Run(loss, dico, false);
                        File.AppendAllLines("loss.csv", new[] { $"{count++}, {currentCost.Get(0)}" });

                        // Update parameters
                        session.Run(optimizer, dico, false);

                        if (Console.KeyAvailable)
                        {
                            break;
                        }
                    }
                } while (!Console.KeyAvailable);

                this._output.Save("MyNetwork");
            }
        }

        private char ToChar(int topn, Volume<float> v, int batch = 0)
        {
            var best = new List<float>();
            for (var i = 0; i < v.Shape.TotalLength / v.Shape.Dimensions[3]; i++)
            {
                best.Add(v.Get(0, 0, i, batch));
            }

            var sorted = best
                .Select((x, i) => new KeyValuePair<float, int>(x, i))
                .OrderBy(x => -x.Key)
                .Take(topn)
                .ToList();

            if (topn == 1)
            {
                return this._data.Vocabulary[sorted[0].Value];
            }

            var sum = sorted.Select(o => o.Key).Sum();
            var rnd = this._random.NextDouble() * sum;

            var j = 0;
            var cumsum = sorted[0].Key;
            while (cumsum < rnd && j < topn - 1)
            {
                j++;
                cumsum += sorted[j].Key;
            }

            return this._data.Vocabulary[sorted[j].Value];
        }
    }
}