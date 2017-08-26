using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.ReinforcementLearning.Deep;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace SlotMachineDemo
{
    public partial class TrainingForm : Form
    {
        #region Member Fields
        private bool _isRunning = false;
        private bool _stopSimThread = false;
        private int _tickCount = 0;
        private Thread _simThread = null;
        private static SlotMachine _slotMachine;
        private static Net<double> _net;
        private static DeepQLearner _brain;
        private const int _NUM_INPUTS = 10; // The slot machine features 3 reels where each reel displays three items, thus this becomes the input to the neural net.
        private static int _numOutputs = Enum.GetNames(typeof(Actions)).Length;
        #endregion

        #region Concstructor
        public TrainingForm()
        {
            InitializeComponent();
            InitSimulation();
        }
        #endregion

        #region UI Events
        /// <summary>
        /// Starts running, resumes or pauses the simulation.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void OnRunResume(object sender, EventArgs e)
        {
            // Run/Pause/Resume
            _isRunning = !_isRunning;           // This affects the simulation thread

            // Update button text
            var button = (sender as Button);
            if(_isRunning)
            {
                button.Text = "Pause";
            }
            else
            {
                button.Text = "Resume";
            }

            // Launch thread if it not exists yet.
            if(_simThread == null)
            {
                _simThread = new Thread(new ThreadStart(SimulationThread));
                _simThread.Start();
            }
        }

        /// <summary>
        /// Stops and resumes the learning of the agent.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void OnStartStopLearning(object sender, EventArgs e)
        {
            // Stop/Resume Learning
            _brain.IsLearning = !_brain.IsLearning;

            // Update button text
            var button = (sender as Button);
            if (_brain.IsLearning)
            {
                button.Text = "Stop Learning";
            }
            else
            {
                button.Text = "Resume Learning";
            }
        }

        private void OnSave(object sender, EventArgs e)
        {

        }

        private void OnLoad(object sender, EventArgs e)
        {

        }

        /// <summary>
        /// Stops the simulation thread.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void OnFormClose(object sender, FormClosedEventArgs e)
        {
            _stopSimThread = true;
        }
        #endregion

        #region Private Functions
        /// <summary>
        /// Processes the simulation and the learning.
        /// </summary>
        private void SimulationThread()
        {
            while (!_stopSimThread)
            {
                if (_isRunning)
                {
                    if (!_slotMachine.IsRunning)
                    {
                        _slotMachine.Start();       // if no reels are running, start the slot machine again
                    }
                    else
                    {
                        double rewardSignal = 0;
                        // Brain Forward to make decision
                        var chosenAction = _brain.Forward(new Volume(GatherInput(), new Shape(GatherInput().Length)));
                        Console.WriteLine("Action: " + (Actions)chosenAction);

                        // Carry out decision
                        if ((Actions)chosenAction == Actions.StopReel)
                        {
                            rewardSignal = _slotMachine.StopReel();
                        }
                        else
                        {
                            // Do nothing, but wait
                            rewardSignal = 0.0;
                        }

                        // Update slot machine and its reels
                        _slotMachine.Tick();

                        // Brain Backward to learn
                        _brain.Backward(rewardSignal);

                        if (!_slotMachine.IsRunning)
                        {
                            // Score is ready
                            Console.WriteLine("score: " + _slotMachine.Score);
                        }

                        SetText(textBoxInformation, "Age: " + _brain.Age + Environment.NewLine + 
                                                  "Experience Replay Size: " + _brain.ExperienceReplaySize + Environment.NewLine +
                                                  "Epsilon: " + _brain.ExplorationEpsilon + Environment.NewLine +
                                                  "Av. Reward: " + _brain.AverageReward + Environment.NewLine +
                                                  "Av. Loss: " + _brain.AverageLoss);
                        //textBoxActions.Text = "";

                        AppendChart(chartAvReward, _tickCount, _brain.AverageReward);
                        AppendChart(chartAvLoss, _brain.Age, _brain.AverageLoss);                        
                    }
                    _tickCount++;
                }
            }
        }

        /// <summary>
        /// Initializes the brain and the slot machine.
        /// </summary>
        private static void InitSimulation()
        {
            // Define input size depending on the number of temporal windows
            int temporalWindow = 0;
            int totalInputCount = _NUM_INPUTS * temporalWindow + _numOutputs * temporalWindow + _NUM_INPUTS;

            // Build Neural Network
            _net = new Net<double>();
            _net.AddLayer(new InputLayer(1, 1, totalInputCount)); // According to the MinimalExample, width and height are set to 1, because the input is not an image.
            _net.AddLayer(new FullyConnLayer(20)); // nodes on the first hidden layer, which undergo the Relu activation
            _net.AddLayer(new ReluLayer());
            _net.AddLayer(new FullyConnLayer(10));
            _net.AddLayer(new ReluLayer());
            _net.AddLayer(new FullyConnLayer(_numOutputs));
            _net.AddLayer(new RegressionLayer());
            var trainer = new SgdTrainer(_net) { LearningRate = 0.01, Momentum = 0.05, BatchSize = 16, L2Decay = 0.001 };

            // Set Training Options to construct the brain (DeepQLearner)
            var trainingOptions = new TrainingOptions()
            {
                Trainer = trainer,
                Net = _net,
                TemporalWindow = temporalWindow,                                // The temporal window specifies how many past states are added to the neural net's input
                ExperienceSize = 30000,                                         // Size of the experience replay memory
                StartLearnThreshold = 1000,                                     // Number of experiences, which shall be present before the learning commences
                LearningStepsTotal = 75000,                                     // Number of iterations for the agent to learn
                LearningStepsBurnin = 500,                                      // Number of actions, which will be chosen randomly right from the start
                QBatchSize = 32,                                                // Number of sampling experiences for training.
                Gamma = 0.9,                                                    // Reward discount
                EpsilonMin = 0.05,                                              // Remaining chance of taking a random action after all learning steps. If 0, decisions are based on the policy only (after all learning steps).
                EpsilonDuringTestTime = 0,                                      // Chance of taking a random action, while the learning process is disabled (agent is put to test).
                ProbabilityActionDistribution = new List<double> { 0.5, 0.5 } // The probability distribution states the chances for an action to be sampled. For example in Flappy Bird, the agent should choose to flap less likely.
            };
            _brain = new DeepQLearner(_NUM_INPUTS, totalInputCount, _numOutputs, trainingOptions);

            // Initialize SlotMachine
            _slotMachine = new SlotMachine();
        }

        /// <summary>
        /// Retrieves information from each reel of the slot machine to output a normalized input vector.
        /// </summary>
        /// <returns>Returns the normalized state of the slot machine.</returns>
        private static double[] GatherInput()
        {
            // Determine the state of the slot machine (how many reels are running/stopped)
            var slotMachineState = 0.0;
            if (_slotMachine.ReelLeft.IsSpinning)
            {
                slotMachineState = 0.25; // Every reel is spinning
            }
            else
            {
                if(_slotMachine.ReelMid.IsSpinning)
                {
                    slotMachineState = 0.5; // Two reels are spinning
                }
                else
                {
                    if(_slotMachine.ReelRight.IsSpinning)
                    {
                        slotMachineState = 0.75; // The last reel is spinning
                    }
                    else
                    {
                        slotMachineState = 1; // No reels are spinning
                    }
                }
            }

            // Get the slot machine's state
            var reelLeftItems = _slotMachine.ReelLeft.SlotItems;
            var reelMidItems = _slotMachine.ReelMid.SlotItems;
            var reelRightItems = _slotMachine.ReelRight.SlotItems;

            // Map inputs for normalization (neural nets work better using values in the range 0 - 1)
            var itemMap = new double[] { 0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0 };

            return new double[]
            {
                slotMachineState,
                itemMap[(int)reelLeftItems[0]],
                itemMap[(int)reelLeftItems[1]],
                itemMap[(int)reelLeftItems[2]],
                itemMap[(int)reelMidItems[0]],
                itemMap[(int)reelMidItems[1]],
                itemMap[(int)reelMidItems[2]],
                itemMap[(int)reelRightItems[0]],
                itemMap[(int)reelRightItems[1]],
                itemMap[(int)reelRightItems[2]],
            };
        }

        private delegate void SetTextCallback(TextBox textBox, string text);
        /// <summary>
        /// Updates a TextBox's text in a thread-safe manner.
        /// </summary>
        /// <param name="textBox">Target TextBox</param>
        /// <param name="text">Text to be set</param>
        private void SetText(TextBox textBox, string text)
        {
            if(textBox.InvokeRequired)
            {
                SetTextCallback d = new SetTextCallback(SetText);
                Invoke(d, new object[] { textBox, text });
            }
            else
            {
                textBox.Text = text;
            }
        }

        private delegate void AppendChartCallback(Chart chart, double x, double y);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="chart"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        private void AppendChart(Chart chart, double x, double y)
        {
            if (chart.InvokeRequired)
            {
                AppendChartCallback d = new AppendChartCallback(AppendChart);
                Invoke(d, new object[] { chart, x, y });
            }
            else
            {
                chart.Series[0].Points.AddXY(x, y);

                if (chart.Series[0].Points.Count % 100 == 0)
                {
                    chart.Refresh();
                }
            }
        }
        #endregion
    }

    /// <summary>
    /// Two action are available for the agent: Wait, StopReel
    /// </summary>
    public enum Actions
    {
        Wait,
        StopReel
    }
}
