using ConvnetSharpOLD;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace DeepQLearning.DRLAgent
{
    /// <summary>
    /// This is a brain object, which receives some inputs and some rewards over time.
    /// Its job is to set the outputs and to maximize the expected reward.
    /// </summary>
    [Serializable]
    public class DeepQLearn
    {
        #region Member Fields
        private TrainingOptions _trainingOptions;

        private int _temporalWindow;
        private int _experienceSize;
        private double _startLearnThreshold;
        private double _gamma;
        private double _learningStepsTotal;
        private double _learningStepsBurnin;
        private double _epsilonMin;
        private double _epsilonTestTime;

        private int _netInputs;
        private int _numStates;
        private int _numActions;
        private int _windowSize;
        private List<Volume> _stateWindow;
        private List<int> _actionWindow;
        private List<double> _rewardWindow;
        private List<double[]> _netWindow;

        private double _age;
        private double _forwardPasses;
        private double _epsilon;
        private double _latestReward;
        private Volume _lastInput;
        private TrainingWindow _averageRewardWindow;
        private TrainingWindow _avergageLossWindow;
        private bool _isLearning;

        private Net _valueNet;
        private Trainer _tdtrainer;

        private Util _util;

        private List<double> _randomActionDistribution;
        private List<Experience> _experienceReplay;
        #endregion

        #region Member Properties
        public bool IsLearning
        {
            get { return _isLearning; }
            set { _isLearning = value; }
        }

        public double EpsilonTestTime
        {
            get { return _epsilonTestTime; }
            set { _epsilonTestTime = value; }
        }
        #endregion

        #region Constructor
        /// <summary>
        /// Initializes the brain according to training parameters and the definition of layers for the neural net.
        /// </summary>
        /// <param name="numStates">For some reason, the number of inputs, w.r.t the temporal window, has to be provided</param>
        /// <param name="numActions">Number of available actions</param>
        /// <param name="opt">Training options</param>
        public DeepQLearn(int numStates, int numActions, TrainingOptions opt)
        {
            _util = new Util();
            _trainingOptions = opt;

            // Initializing members, overriding defaults of no custom values were set
            // in number of time steps, of temporal memory (past states)
            // the ACTUAL input to the net will be (x,a) temporalWindow times, and followed by current x
            // so to have no information from previous time step going into value function, set to 0.
            _temporalWindow = opt.temporalWindow != int.MinValue ? opt.temporalWindow : 0;
            // size of experience replay memory
            _experienceSize = opt.experienceSize != int.MinValue ? opt.experienceSize : 30000;
            // number of examples in experience replay memory before we begin learning
            _startLearnThreshold = opt.startLearnThreshold != double.MinValue ? opt.startLearnThreshold : Math.Floor(Math.Min(_experienceSize * 0.1, 1000));
            // gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
            // discount value for the reward
            _gamma = opt.gamma != double.MinValue ? opt.gamma : 0.8;

            // number of steps we will learn for
            _learningStepsTotal = opt.learningStepsTotal != int.MinValue ? opt.learningStepsTotal : 100000;
            // how many steps of the above to perform only random actions (in the beginning)?
            _learningStepsBurnin = opt.learningStepsBurnin != int.MinValue ? opt.learningStepsBurnin : 3000;
            // what epsilon value do we bottom out on? 0.0 => purely deterministic policy at end
            _epsilonMin = opt.epsilonMin != double.MinValue ? opt.epsilonMin : 0.05;
            // what epsilon to use at test time? (i.e. when learning is disabled)
            _epsilonTestTime = opt.epsilonTestTime != double.MinValue ? opt.epsilonTestTime : 0.00;

            // advanced feature: Sometimes a random action should be biased towards some values
            // for example in flappy bird, we may want to choose to not flap more often
            if (opt.radmomActionDistribution != null)
            {
                // this better sum to 1 by the way, and be of length numActions
                _randomActionDistribution = opt.radmomActionDistribution;
                if (_randomActionDistribution.Count != numActions)
                {
                    Console.WriteLine("TROUBLE. randomActionDistribution should be same length as numActions.");
                }

                var sumOfDistance = _randomActionDistribution.Sum();
                if (Math.Abs(sumOfDistance - 1.0) > 0.0001) { Console.WriteLine("TROUBLE. randomActionDistribution should sum to 1!"); }
            }
            else
            {
                _randomActionDistribution = new List<double>();
            }

            // states that go into neural net to predict optimal action look as
            // x0,a0,x1,a1,x2,a2,...xt
            // this variable controls the size of that temporalWindow. 
            // Actions are encoded as 1-of-k hot vectors
            _netInputs = numStates * _temporalWindow + numActions * _temporalWindow + numStates;
            _numStates = numStates;
            _numActions = numActions;
            _windowSize = Math.Max(_temporalWindow, 2); // must be at least 2, but if we want more context even more
            _stateWindow = new List<Volume>();
            _actionWindow = new List<int>();
            _rewardWindow = new List<double>();
            _netWindow = new List<double[]>();

            // Init windows using dummy data
            for (int i = 0; i < _windowSize; i++)
            {
                _stateWindow.Add(new Volume(1, 1, 1));
                _actionWindow.Add(0);
                _rewardWindow.Add(0.0);
                _netWindow.Add(new double[] { 0.0 });
            }

            // create [state -> value of all possible actions] modeling net for the value function
            var layerDefinitions = new List<LayerDefinition>();

            if (opt.layerDefinitions != null)
            {
                // this is an advanced usage feature, because size of the input to the network, and number of
                // actions must check out. This is not very pretty Object Oriented programming but I can't see
                // a way out of it :(
                layerDefinitions = opt.layerDefinitions;
                if (layerDefinitions.Count < 2) { Console.WriteLine("TROUBLE! must have at least 2 layers"); }
                if (layerDefinitions[0].type != "input") { Console.WriteLine("TROUBLE! first layer must be input layer!"); }
                if (layerDefinitions[layerDefinitions.Count - 1].type != "regression") { Console.WriteLine("TROUBLE! last layer must be input regression!"); }
                if (layerDefinitions[0].out_depth * layerDefinitions[0].out_sx * layerDefinitions[0].out_sy != _netInputs)
                {
                    Console.WriteLine("TROUBLE! Number of inputs must be num_states * temporal_window + num_actions * temporal_window + num_states!");
                }
                if (layerDefinitions[layerDefinitions.Count - 1].num_neurons != _numActions)
                {
                    Console.WriteLine("TROUBLE! Number of regression neurons should be num_actions!");
                }
            }
            else
            {
                // create a very simple neural net by default
                layerDefinitions.Add(new LayerDefinition { type = "input", out_sx = 1, out_sy = 1, out_depth = _netInputs });
                if (opt.hiddenLayerSizes != null)
                {
                    // allow user to specify this via the option, for convenience
                    var hl = opt.hiddenLayerSizes;
                    for (var k = 0; k < hl.Length; k++)
                    {
                        layerDefinitions.Add(new LayerDefinition { type = "fc", num_neurons = hl[k], activation = "relu" }); // relu by default
                    }
                }
            }

            // Create the network
            _valueNet = new Net();
            _valueNet.makeLayers(layerDefinitions);

            // and finally we need a Temporal Difference Learning trainer!
            var options = new Options { learningRate = 0.005, momentum = 0.001, batchSize = 128, l2_decay = 0.01, l1_decay = 0.01 };
            if (opt.options != null)
            {
                options = opt.options; // allows user to overwrite this
            }

            _tdtrainer = new Trainer(_valueNet, options);

            // experience replay
            _experienceReplay = new List<Experience>();

            // various housekeeping variables
            _age = 0; // incremented every backward()
            _forwardPasses = 0; // incremented every forward()
            _epsilon = 1.0; // controls exploration exploitation tradeoff. Should be annealed over time
            _latestReward = 0;
            _averageRewardWindow = new TrainingWindow(1000, 10);
            _avergageLossWindow = new TrainingWindow(1000, 10);
            _isLearning = true;
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Forward is used to gather input information, which are then fed to the neural net to make a decision for the current situation.
        /// </summary>
        /// <param name="inputVolume">Current state input of the agent</param>
        /// <returns></returns>
        public int Forward(Volume inputVolume)
        {
            // compute forward (behavior) pass given the input neuron signals from the agent
            _forwardPasses++;
            _lastInput = inputVolume; // back this up

            // create network input
            int action;
            double[] netInput;
            if (_forwardPasses > _temporalWindow)
            {
                // we have enough to actually do something reasonable
                netInput = GetNetInput(inputVolume);
                if (_isLearning)
                {
                    // compute epsilon for the epsilon-greedy policy
                    _epsilon = Math.Min(1.0, Math.Max(_epsilonMin, 1.0 - (_age - _learningStepsBurnin) / (_learningStepsTotal - _learningStepsBurnin)));
                }
                else
                {
                    _epsilon = _epsilonTestTime; // use test-time value
                }

                var rf = _util.Randf(0, 1);
                if (rf < _epsilon)
                {
                    // choose a random action with epsilon probability
                    action = RandomAction();
                }
                else
                {
                    // otherwise use our policy to make decision
                    var maxact = Policy(netInput, true);
                    action = maxact.action;
                }
            }
            else
            {
                // pathological case that happens first few iterations 
                // before we accumulate _windowSize inputs
                netInput = new List<double>().ToArray();
                action = RandomAction();
            }

            // remember the state and action we took for backward pass
            _netWindow.RemoveAt(0);
            _netWindow.Add(netInput);
            _stateWindow.RemoveAt(0);
            _stateWindow.Add(inputVolume);
            _actionWindow.RemoveAt(0);
            _actionWindow.Add(action);

            return action;
        }

        /// <summary>
        /// Triggers the training process and tracks the rewards.
        /// </summary>
        /// <param name="reward">Gathered reward of the agent</param>
        public void Backward(double reward)
        {
            _latestReward = reward;
            _averageRewardWindow.add(reward);

            _rewardWindow.RemoveAt(0);
            _rewardWindow.Add(reward);

            if (!_isLearning) { return; } // if the agent is not learning, then skip the learning logic

            // various book-keeping
            _age++;

            // save new experience given initial input state, output action, received reward and final state
            // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
            // (given that an appropriate number of state measurements already exist, of course)
            if (_forwardPasses > _temporalWindow + 1)
            {
                var e = new Experience();
                var n = _windowSize;
                e.initialState = _netWindow[n - 2];
                e.initialAction = _actionWindow[n - 2];
                e.initialReward = _rewardWindow[n - 2];
                e.finalState = _netWindow[n - 1];

                if (_experienceReplay.Count < _experienceSize)
                {
                    _experienceReplay.Add(e);
                }
                else
                {
                    // replace. finite memory!
                    var randomInt = _util.RandInt(0, _experienceSize);
                    _experienceReplay[randomInt] = e;
                }
            }

            // learn based on experience, once we have some samples to go on
            // this is where the magic happens...
            if (_experienceReplay.Count > _startLearnThreshold)
            {
                var averageLoss = 0.0;
                for (var k = 0; k < _tdtrainer.batchSize; k++)
                {
                    var re = _util.RandInt(0, _experienceReplay.Count);
                    var exp = _experienceReplay[re];
                    var initialInputVolume = new Volume(1, 1, _netInputs);
                    initialInputVolume.w = exp.initialState;
                    var maxact = Policy(exp.finalState, false); // sounds conflicting to the idea, that there is a q-value for each action
                    var r = exp.initialReward + _gamma * maxact.value; // potential reward? Aren't we supposed to add more rewards times gamma^t ?

                    var ystruct = new Entry { dim=exp.initialAction, val=r};
                    var loss = _tdtrainer.train(initialInputVolume, ystruct); // is this just the plain training of the neural net??
                    averageLoss += double.Parse(loss["loss"]);
                }

                averageLoss = averageLoss / _tdtrainer.batchSize;
                _avergageLossWindow.add(averageLoss);
            }
        }

        /// <summary>
        /// Outputs a string for logging some data.
        /// </summary>
        /// <returns></returns>
        public string VisSelf()
        {
            var t = "";
            t += "experience replay size: " + _experienceReplay.Count + Environment.NewLine;
            t += "exploration epsilon: " + _epsilon + Environment.NewLine;
            t += "age: " + _age + Environment.NewLine;
            t += "average Q-learning loss: " + _avergageLossWindow.get_average() + Environment.NewLine;
            t += "smooth-ish reward: " + _averageRewardWindow.get_average() + Environment.NewLine;

            return t;
        }
        #endregion

        #region Local Functions
        /// <summary>
        /// Returns a random action.
        /// </summary>
        /// <returns></returns>
        private int RandomAction()
        {
            // a bit of a helper function. It returns a random action
            // we are abstracting this away because in future we may want to 
            // do more sophisticated things. For example some actions could be more
            // or less likely at "rest"/default state.

            int action = _util.RandInt(0, _numActions);

            if (_randomActionDistribution.Count != 0)
            {
                // okay, lets do some fancier sampling:
                var p = _util.Randf(0, 1.0);
                var cumprob = 0.0;
                for (var k = 0; k < _numActions; k++)
                {
                    cumprob += _randomActionDistribution[k];
                    if (p < cumprob) { action = k; break; }
                }
            }

            return action;
        }

        /// <summary>
        /// Uses the neural network to find the best policy made of action paired to its q-value.
        /// </summary>
        /// <param name="state">Input state, which is fed to the neural network</param>
        /// <returns>Returns ana ction along with its q-value based on the input state</returns>
        private Action Policy(double[] state, bool forward)
        {
            // compute the value of doing any action in this state
            // and return the argmax action and its value
            var stateVolume = new Volume(1, 1, _netInputs);
            stateVolume.w = state;
            var actionValues = _valueNet.forward(stateVolume, false);
            var maxk = 0;
            var maxval = actionValues.w[0];
            for (var k = 1; k < _numActions; k++)
            {
                if (actionValues.w[k] > maxval) { maxk = k; maxval = actionValues.w[k]; }
            }

            if(forward)
            {
                string[] arrayIn = Array.ConvertAll(stateVolume.w, x => x.ToString());
                Console.WriteLine("in " + String.Join(",", arrayIn.Select(p => p.ToString()).ToArray()));
                //string[] arrayOut = Array.ConvertAll(actionValues.w, x => x.ToString());
                //Console.WriteLine("out " + String.Join(",", arrayOut.Select(p => p.ToString()).ToArray()));
            }

            return new Action { action = maxk, value = maxval };
        }

        /// <summary>
        /// Gathers information from the past (inputs and actions) and the presence (inputs only) to create an input vector of that information.
        /// The past information relies on the size of the temporal window.
        /// </summary>
        /// <param name="currentInputVolume">Current input information</param>
        /// <returns>Returns the input vector based on the present and past information.</returns>
        private double[] GetNetInput(Volume currentInputVolume)
        {
            // return s = (x,a,x,a,x,a,xt) state vector.        xt = currentInputVolume
            // It's a concatenation of last _windowSize (x,a) pairs and current state xt
            List<double> w = new List<double>();

            // start with current state and now go backwards and append states and actions from history _temporalWindow times
            w.AddRange(currentInputVolume.w);

            var n = _windowSize;
            for (var k = 0; k < _temporalWindow; k++)
            {
                // state
                w.AddRange(_stateWindow[n - 1 - k].w);
                // action, encoded as 1-of-k indicator vector. We scale it up a bit because
                // we dont want weight regularization to undervalue this information, as it only exists once
                var action1ofk = new double[_numActions];
                for (var q = 0; q < _numActions; q++) action1ofk[q] = 0.0;
                action1ofk[_actionWindow[n - 1 - k]] = 1.0 * _numStates;
                w.AddRange(action1ofk);
            }
            return w.ToArray();
        }
        #endregion
    }
}
