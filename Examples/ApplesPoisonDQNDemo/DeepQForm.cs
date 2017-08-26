using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.ReinforcementLearning.Deep;
using ConvNetSharp.Volume;

using DeepQLearning.DRLAgent;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading;
using System.Windows.Forms;

namespace DeepQLearning
{
    public partial class DeepQForm : Form
    {
        #region Member Fields
        private Thread _workerThread = null;
        private Boolean _needToStop = false, _isPaused = false;
        private QAgent _qAgent;
        private int _interval = 30;
        private string _netFile = Environment.CurrentDirectory + "\\deepQnet.dat";
        #endregion

        #region Constructor
        public DeepQForm()
        {
            InitializeComponent();

            // Fix Panel double buffering issue
            typeof(Panel).InvokeMember("DoubleBuffered",
            BindingFlags.SetProperty | BindingFlags.Instance | BindingFlags.NonPublic,
            null, canvas, new object[] { true });
        }
        #endregion

        private void PaintCanvas(object sender, PaintEventArgs e)
        {
            if (_qAgent != null)
            {
                displayBox.Text = _qAgent.DrawWorld(e.Graphics);

                switch (_qAgent.SimulationSpeed)
                {
                    case 0:
                        displayBox.Text += Environment.NewLine + "Simulation speed: Slow";
                        break;

                    case 1:
                        displayBox.Text += Environment.NewLine + "Simulation speed: Normal";
                        break;

                    case 2:
                        displayBox.Text += Environment.NewLine + "Simulation speed: Fast";
                        break;

                    case 3:
                        displayBox.Text += Environment.NewLine + "Simulation speed: Very Fast";
                        break;
                }
            }

            canvas.Update();
        }

        #region UI Events
        private void OnStopLearning(object sender, EventArgs e)
        {
            if (_qAgent != null)
            {
                _qAgent.StopLearn();
            }
        }

        private void OnStartLearning(object sender, EventArgs e)
        {
            if (_qAgent == null)
            {
                var numInputs = 27; // 9 eyes, each sees 3 numbers (wall, green, red thing proximity)
                var numActions = 5; // 5 possible angles agent can turn
                var temporalWindow = 0; // amount of temporal memory. 0 = agent lives in-the-moment :)
                var networkSize = numInputs * temporalWindow + numActions * temporalWindow + numInputs;

                // Build Neural Network
                var net = new Net<double>();
                net.AddLayer(new InputLayer(1, 1, networkSize)); // According to the MinimalExample, width and height are set to 1, because the input is not an image.
                net.AddLayer(new FullyConnLayer(40)); // nodes on the first hidden layer, which undergo the Relu activation
                net.AddLayer(new ReluLayer());
                net.AddLayer(new FullyConnLayer(40));
                net.AddLayer(new ReluLayer());
                net.AddLayer(new FullyConnLayer(20));
                net.AddLayer(new ReluLayer());
                net.AddLayer(new FullyConnLayer(numActions));
                net.AddLayer(new RegressionLayer());
                var trainer = new SgdTrainer(net) { LearningRate = 0.005, Momentum = 0.01, BatchSize = 128, L2Decay = 0.001, L1Decay = 0.001 };

                // Set Training Options to construct the brain (DeepQLearner)
                var trainingOptions = new TrainingOptions()
                {
                    Trainer = trainer,
                    Net = net,
                    TemporalWindow = temporalWindow,                                    // The temporal window specifies how many past states are added to the neural net's input
                    ExperienceSize = 50000,                                             // Size of the experience replay memory
                    StartLearnThreshold = 4000,                                         // Number of experiences, which shall be present before the learning commences
                    LearningStepsTotal = 500000,                                        // Number of iterations for the agent to learn
                    LearningStepsBurnin = 3000,                                         // Number of actions, which will be chosen randomly right from the start
                    QBatchSize = 128,                                                   // Number of sampling experiences for training.
                    Gamma = 0.9,                                                        // Reward discount
                    EpsilonMin = 0.05,                                                  // Remaining chance of taking a random action after all learning steps. If 0, decisions are based on the policy only (after all learning steps).
                    EpsilonDuringTestTime = 0.05,                                       // Chance of taking a random action, while the learning process is disabled (agent is put to test).
                    ProbabilityActionDistribution = null                                // The probability distribution states the chances for an action to be sampled. For example in Flappy Bird, the agent should choose to flap less likely.
                };

                var brain = new DeepQLearner(numInputs, networkSize, numActions, trainingOptions);
                _qAgent = new QAgent(brain, canvas.Width, canvas.Height);
            }
            else
                _qAgent.StartLearn();

            if (_workerThread == null)
            {
                _workerThread = new Thread(new ThreadStart(BackgroundThread));
                _workerThread.Start();
            }
        }

        private void OnPause(object sender, EventArgs e)
        {
            if (_isPaused)
            {
                pauseButton.Text = "Pause";
                _isPaused = false;
            }
            else
            {
                pauseButton.Text = "Continue";
                _isPaused = true;
            }
        }

        private void OnSaveNet(object sender, EventArgs e)
        {
            // Save the netwok to file
            using (FileStream fstream = new FileStream(_netFile, FileMode.Create))
            {
                new BinaryFormatter().Serialize(fstream, _qAgent);
            }

            displayBox.Text = "QNetwork saved successfully";
        }

        private void OnLoadNet(object sender, EventArgs e)
        {
            // Load the netwok from file
            using (FileStream fstream = new FileStream(_netFile, FileMode.Open))
            {
                _qAgent = new BinaryFormatter().Deserialize(fstream) as QAgent;
                _qAgent.Reinitialize();
            }

            if (_workerThread == null)
            {
                _workerThread = new Thread(new ThreadStart(BackgroundThread));
                _workerThread.Start();
            }
        }

        private void OnFormClose(object sender, FormClosedEventArgs e)
        {
            _needToStop = true;

            if (_workerThread != null)
            {
                // stop worker thread
                _needToStop = true;
                while (!_workerThread.Join(100))
                    Application.DoEvents();
                _workerThread = null;
            }
        }

        private void OnNormalSpeed(object sender, EventArgs e)
        {
            if (_qAgent != null)
            {
                _qAgent.RunNormal();
                _interval = 25;
            }
        }

        private void OnFastSpeed(object sender, EventArgs e)
        {
            if (_qAgent != null)
            {
                _qAgent.RunFast();
                _interval = 10;
            }
        }

        private void OnVeryFastSpeed(object sender, EventArgs e)
        {
            if (_qAgent != null)
            {
                _qAgent.RunVeryFast();
                _interval = 0;
            }
        }

        private void OnSlowSpeed(object sender, EventArgs e)
        {
            if (_qAgent != null)
            {
                _qAgent.RunSlow();
                _interval = 50;
            }
        }
        #endregion

        // Delegates to enable async calls for setting controls properties
        private delegate void UpdateUICallback(Panel panel);

        // Thread safe updating of UI
        private void UpdateUI(Panel panel)
        {
            if (_needToStop)
                return;

            if (panel.InvokeRequired)
            {
                UpdateUICallback d = new UpdateUICallback(UpdateUI);
                Invoke(d, new object[] { panel });
            }
            else
            {
                panel.Refresh();
            }
        }

        private void BackgroundThread()
        {
            while (!_needToStop)
            {
                if (!_isPaused)
                {
                    _qAgent.Tick();
                    UpdateUI(canvas);
                }

                Thread.Sleep(_interval);
            }
        }
    }
}