using ConvNetSharp.Core;
using ConvNetSharp.ReinforcementLearning.Deep.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvNetSharp.Volume.Double;
using ConvNetSharp.Volume;

namespace ConvNetSharp.ReinforcementLearning.Deep
{
    public class DeepQLearner
    {
        #region Member Fields
        private bool _isLearning = true;
        private TrainingOptions _trainingOptions;
        private int _numInputs;
        private int _numActions;
        private int _netInputs;
        private int _windowSize;
        private Random _random = new Random();

        private int _age = 0;
        private double _forwardPasses = 0;
        private double _epsilon = 1.0;
        private double _latestReward = 0;
        private Volume.Double.Volume _lastInput;
        private List<Experience> _experienceReplay;

        // Windows
        private List<Volume.Double.Volume> _stateWindow;
        private List<int> _actionWindow;
        private List<double> _rewardWindow;
        private List<Volume.Double.Volume> _netWindow;
        private TrainingWindow _averageRewardWindow;
        private TrainingWindow _averageLossWindow;
        #endregion

        #region Member Properties
        public bool IsLearning
        {
            get { return _isLearning; }
            set { _isLearning = value; }
        }

        public int Age
        {
            get { return _age; }
        }

        public double AverageLoss
        {
            get { return _averageLossWindow.GetAverage(); }
        }

        public double AverageReward
        {
            get { return _averageRewardWindow.GetAverage(); }
        }

        public double ExplorationEpsilon
        {
            get { return _epsilon; }
        }

        public int ExperienceReplaySize
        {
            get { return _experienceReplay.Count; }
        }
        #endregion

        #region Constructor
        /// <summary>
        /// Initializes a DeepQLearner featuring Deep Reinforcement Learning.
        /// </summary>
        /// <param name="numInputs">Number of inputs</param>
        /// <param name="numTotalInputs">Number of total inputs</param>
        /// <param name="numActions">Number of actions</param>
        /// <param name="options">Training options including the neural net and its trainer.</param>
        public DeepQLearner(int numInputs, int numTotalInputs, int numActions, TrainingOptions options)
        {
            // Initialize and assign members
            _trainingOptions = options;
            _numInputs = numInputs;
            _numActions = numActions;
            _netInputs = numTotalInputs;
            _experienceReplay = new List<Experience>();
            _averageRewardWindow = new TrainingWindow(1000, 10);
            _averageLossWindow = new TrainingWindow(1000, 10);

            // Verify probability distribution
            if (_trainingOptions.ProbabilityActionDistribution != null)
            {
                if(_trainingOptions.ProbabilityActionDistribution.Count != _numActions)
                {
                    throw (new Exception("The count of the probability distribution has to match the number of actions."));
                }

                if (Math.Abs(_trainingOptions.ProbabilityActionDistribution.Sum() - 1.0) > 0.0001)
                {
                    throw (new Exception("The values of the probability distribution should sum to 1."));
                }
            }
            else
            {
                _trainingOptions.ProbabilityActionDistribution = new List<double>(); // An empty list will imply the usage of a uniform probability distribution
            }

            // Initialize windows
            _windowSize = Math.Max(_trainingOptions.TemporalWindow, 2); // must be at least 2, but if more context is desired, add more
            _stateWindow = new List<Volume.Double.Volume>();            // Single states
            _actionWindow = new List<int>();
            _rewardWindow = new List<double>();
            _netWindow = new List<Volume.Double.Volume>();              // Convolute of states which are actually fed to the neural net

            // Add dummy data to the windows
            for (int i = 0; i < _windowSize; i++)
            {
                _stateWindow.Add(new Volume.Double.Volume(new double[numActions], new Shape(numActions)));
                _actionWindow.Add(0);
                _rewardWindow.Add(0.0);
                _netWindow.Add(new Volume.Double.Volume(new double[numActions], new Shape(numActions)));
            }

            // TODO: Verfiy neural net composition
            //_trainingOptions.Net.Layers.
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Forward is used to gather input information, which are then fed to the neural net to make a decision for the current situation.
        /// </summary>
        /// <param name="inputVolume">Current state input of the agent</param>
        /// <returns>Returns the decision for an action, based on the learned policy or the epsilon-greedy policy (exploration).</returns>
        public int Forward(Volume.Double.Volume inputVolume)
        {
            _forwardPasses++;
            _lastInput = inputVolume;
            int action = 0;

            // Create net input
            Volume.Double.Volume netInput;

            // Check if enough data has been gathered
            if(_forwardPasses > _trainingOptions.TemporalWindow)
            {
                // Gather input information (depending on the temporal window)
                netInput = GetNetInput(inputVolume);

                // Determine/Update epsilon
                if(_isLearning)
                {
                    // compute epsilon for the epsilon-greedy policy
                    _epsilon = Math.Min(1.0, Math.Max(_trainingOptions.EpsilonMin,
                                    1.0 - ((double)_age - _trainingOptions.LearningStepsBurnin) /
                                    (_trainingOptions.LearningStepsTotal - _trainingOptions.LearningStepsBurnin)));
                }
                else
                {
                    _epsilon = _trainingOptions.EpsilonDuringTestTime; // use test time value
                }

                // Make decision
                var randomDouble = _random.NextDouble();
                // Use random action
                if (randomDouble < _epsilon)
                {
                    action = RandomAction();
                }
                // Use Policy
                else
                {
                    var outputVolume = _trainingOptions.Net.Forward(netInput);
                    var actionPolicy = RetrievePolicy(outputVolume.ToArray());
                    action = actionPolicy.Action;
                }
            }
            else
            {
                // Pathological case that happens first few iterations 
                // Accumulate _windowSize inputs and actions
                netInput = new Volume.Double.Volume(new double[_numActions], new Shape(_numActions));
                action = RandomAction();
            }

            // Save the state and the chosen action
            _netWindow.RemoveAt(0);
            _netWindow.Add(netInput);
            _stateWindow.RemoveAt(0);
            _stateWindow.Add(inputVolume);
            _actionWindow.RemoveAt(0);
            _actionWindow.Add(action);

            return action;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reward"></param>
        public void Backward(double reward)
        {
            _latestReward = reward;
            _averageRewardWindow.Add(reward);

            _rewardWindow.RemoveAt(0);
            _rewardWindow.Add(reward);

            if (!_isLearning)
            {
                return; // if the agent is not learning, then skip the learning logic
            }

            _age++;

            // Save the new experience given initial input state, output action, received reward and final state
            // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
            // (given that an appropriate number of state measurements already exist, of course)
            if (_forwardPasses > _trainingOptions.TemporalWindow + 1)
            {
                var newExperience = new Experience(_netWindow[_windowSize - 2], _actionWindow[_windowSize - 2], _rewardWindow[_windowSize - 2], _netWindow[_windowSize - 1]);

                if(_experienceReplay.Count < _trainingOptions.ExperienceSize)
                {
                    _experienceReplay.Add(newExperience);
                }
                else
                {
                    // replace a randomly selected experience, if the experience replay memory is full
                    var randomInt = _random.Next(0, _trainingOptions.ExperienceSize);
                    _experienceReplay[randomInt] = newExperience;
                }
            }

            // Learn based on experience, once enough samples are available
            if(_experienceReplay.Count > _trainingOptions.StartLearnThreshold)
            {
                var averageLoss = 0.0;
                for(int i = 0; i < _trainingOptions.QBatchSize; i++)
                {
                    // Sample random experience
                    var experience = _experienceReplay[_random.Next(0, _experienceReplay.Count)];

                    // Compute new action value
                    var outputVolume = _trainingOptions.Net.Forward(experience.FinalState);
                    var actionPolicy = RetrievePolicy(outputVolume.ToArray());
                    var newActionValue = experience.Reward + _trainingOptions.Gamma * actionPolicy.Value;

                    // Create desired output volume
                    var outputArrayUpdated = outputVolume.ToArray();
                    outputArrayUpdated[actionPolicy.Action] = newActionValue;
                    var desiredOutputVolume = new Volume.Double.Volume(outputArrayUpdated, new Shape(outputArrayUpdated.Length));
                    
                    // Train input/output pair
                    _trainingOptions.Trainer.Train(experience.InitialState, desiredOutputVolume);
                    averageLoss += _trainingOptions.Trainer.Loss;
                }

                averageLoss = averageLoss / _trainingOptions.QBatchSize;
                _averageLossWindow.Add(averageLoss);
            }
        }
        #endregion

        #region Private Functions
        /// <summary>
        /// Samples an action from a uniform distribution or a specified probability distribution.
        /// </summary>
        /// <returns>Returns a random action.</returns>
        private int RandomAction()
        {
            // Pick an action based on a uniform distribution
            int action = _random.Next(0, _numActions);

            // Sample an action from a given probability distribution
            if(_trainingOptions.ProbabilityActionDistribution.Count != 0)
            {
                var randomDouble = _random.NextDouble();
                var cumulatedProbability = 0.0;
                for(int i = 0; i < _numActions; i++)
                {
                    cumulatedProbability += _trainingOptions.ProbabilityActionDistribution[i];
                    if(randomDouble < cumulatedProbability)
                    {
                        action = i;
                        break;
                    }
                }
            }

            return action;
        }

        /// <summary>
        /// Uses the neural network to find the best policy for an action paired to its q-value.
        /// </summary>
        /// <param name="actionValues">Output array of the neural network, which contains all action values</param>
        /// <returns>Returns an action along with its q-value based on the input state</returns>
        private ActionValuePolicy RetrievePolicy(double[] actionValues)
        {
            // Find highest action value, i.e. policy
            var maxAction = 0;
            var maxValue = actionValues[0];
            for(int i = 1; i < actionValues.Length; i++)
            {
                if(actionValues[i] > maxValue)
                {
                    maxAction = i;
                    maxValue = actionValues[i];
                }
            }
            
            return new ActionValuePolicy() {Action = maxAction, Value = maxValue};
        }

        /// <summary>
        /// Gathers information from the past (inputs and actions) and the presence (inputs only) to create an input vector of that information.
        /// The past information relies on the size of the temporal window.
        /// </summary>
        /// <param name="currentInputVolume">Current input information</param>
        /// <returns>Returns the input volume based on the present and past inputs.</returns>
        private Volume.Double.Volume GetNetInput(Volume.Double.Volume currentInputVolume)
        {
            // return s = (x,a,x,a,x,a,xt) state vector.        xt = currentInputVolume
            // It's a concatenation of last _windowSize (x,a) pairs and current state xt

            var n = _windowSize;
            for (var k = 0; k < _trainingOptions.TemporalWindow; k++)
            {
                // state
                currentInputVolume.Add(_stateWindow[n - 1 - k]);
                // action, encoded as 1-of-k indicator vector. We scale it up a bit because
                // we dont want weight regularization to undervalue this information, as it only exists once
                var action1ofk = new double[_numActions];
                for (var q = 0; q < _numActions; q++)
                {
                    action1ofk[q] = 0.0;
                }
                action1ofk[_actionWindow[n - 1 - k]] = 1.0 * _numInputs;
                currentInputVolume.Add(action1ofk);
            }
            return currentInputVolume;
        }
        #endregion
    }
}