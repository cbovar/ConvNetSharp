using ConvNetSharp.Core;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Core.Training.Double;
using System.Collections.Generic;

namespace ConvNetSharp.ReinforcementLearning.Deep
{
    /// <summary>
    /// This object stores all relevant hyperparameters for training.
    /// </summary>
    public class TrainingOptions
    {
        #region Member Fields
        // Neural net members
        private SgdTrainer _trainer;
        private Net<double> _net;

        // Reinforcement learning member variables
        // Explanations can be found inside the "Member Properties" region
        private int _temporalWindow = 0;
        private int _experienceSize = 1000;
        private int _startLearnThreshold = 200;
        private int _learningStepsTotal = 5000;
        private int _learningStepsBurnin = 100;
        private int _qBatchSize = 1;
        private double _gamma = 0.95;
        private double _epsilonMin = 0.05;
        private double _epsilonTestDuration = 0;
        private List<double> _probabilityActionDistribution = new List<double>();
        #endregion

        #region Member Properties
        /// <summary>
        /// Stochastic Gradient Descent trainer for the neural net.
        /// </summary>
        public SgdTrainer Trainer
        {
            get { return _trainer; }
            set { _trainer = value; }
        }

        /// <summary>
        /// Neural network.
        /// </summary>
        public Net<double> Net
        {
            get { return _net; }
            set { _net = value; }
        }

        /// <summary>
        /// The temporal window specifies how many past states are added to the neural net's input.
        /// </summary>
        public int TemporalWindow
        {
            get { return _temporalWindow; }
            set { _temporalWindow = value; }
        }

        /// <summary>
        /// Size of the experience replay memory.
        /// </summary>
        public int ExperienceSize
        {
            get { return _experienceSize; }
            set { _experienceSize = value; }
        }

        /// <summary>
        /// Number of experiences, which shall be present before the learning commences.
        /// </summary>
        public int StartLearnThreshold
        {
            get { return _startLearnThreshold; }
            set { _startLearnThreshold = value; }
        }

        /// <summary>
        /// Number of iterations for the agent to learn (related to the epsilon-greey exploration).
        /// </summary>
        public int LearningStepsTotal
        {
            get { return _learningStepsTotal; }
            set { _learningStepsTotal = value; }
        }

        /// <summary>
        /// Number of actions, which will be chosen randomly right from the start.
        /// </summary>
        public int LearningStepsBurnin
        {
            get { return _learningStepsBurnin; }
            set { _learningStepsBurnin = value; }
        }

        /// <summary>
        /// Number of sampling experiences for training.
        /// </summary>
        public int QBatchSize
        {
            get { return _qBatchSize; }
            set { _qBatchSize = value; }
        }

        /// <summary>
        /// Reward discount.
        /// </summary>
        public double Gamma
        {
            get { return _gamma; }
            set { _gamma = value; }
        }

        /// <summary>
        /// Remaining chance of taking a random action after all learning steps. If 0, decisions are based on the policy only (after all learning steps).
        /// </summary>
        public double EpsilonMin
        {
            get { return _epsilonMin; }
            set { _epsilonMin = value; }
        }

        /// <summary>
        /// Chance of taking a random action, while the learning process is disabled (agent is put to test).
        /// </summary>
        public double EpsilonDuringTestTime
        {
            get { return _epsilonTestDuration; }
            set { _epsilonTestDuration = value; }
        }

        /// <summary>
        /// The probability distribution states the chances for an action to be sampled. For example in Flappy Bird, the agent should choose to flap less likely.
        /// </summary>
        public List<double> ProbabilityActionDistribution
        {
            get { return _probabilityActionDistribution; }
            set { _probabilityActionDistribution = value; }
        }
        #endregion

        #region Constructor
        public TrainingOptions() { }
        #endregion
    }
}