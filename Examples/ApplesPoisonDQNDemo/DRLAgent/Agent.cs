using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.ReinforcementLearning.Deep;
using ConvNetSharp.Volume;
using System;
using System.Collections.Generic;

namespace DeepQLearning.DRLAgent
{
    // A single agent
    [Serializable]
    public class Agent
    {
        #region Member Fields
        public List<Eye> eyes;
        public List<double[]> actions;
        public double angle, oangle, rewardBonus, digestionSignal;
        public double rad, rotationSpeed1, rotationSpeed2, prevactionix;
        public Vector p, op;
        public int actionIndex;
        public DeepQLearner brain;
        #endregion

        #region Constructor
        public Agent(DeepQLearner brain)
        {
            this.brain = brain;

            // positional information
            this.p = new Vector(50, 50);
            this.op = this.p; // old position
            this.angle = 0; // direction facing

            this.actions = new List<double[]>();
            this.actions.Add(new double[] { 1, 1 });
            this.actions.Add(new double[] { 0.8, 1 });
            this.actions.Add(new double[] { 1, 0.8 });
            this.actions.Add(new double[] { 0.5, 0 });
            this.actions.Add(new double[] { 0, 0.5 });

            // properties
            this.rad = 10;
            this.eyes = new List<Eye>();
            for (var k = 0; k < 9; k++) { this.eyes.Add(new Eye((k - 3) * 0.25)); }

            this.rewardBonus = 0.0;
            this.digestionSignal = 0.0;

            // outputs on world
            this.rotationSpeed1 = 0.0; // rotation speed of 1st wheel
            this.rotationSpeed2 = 0.0; // rotation speed of 2nd wheel

            this.prevactionix = -1;
        }
        #endregion

        #region Public Functions
        public void Forward()
        {
            // in forward pass the agent simply behaves in the environment
            // create input to brain
            var numEyes = eyes.Count;
            var inputArray = new double[numEyes * 3];
            for (var i = 0; i < numEyes; i++)
            {
                var eyes = this.eyes[i];
                inputArray[i * 3] = 1.0;
                inputArray[i * 3 + 1] = 1.0;
                inputArray[i * 3 + 2] = 1.0;
                if (eyes.sensedType != -1)
                {
                    // sensed_type is 0 for wall, 1 for food and 2 for poison.
                    // lets do a 1-of-k encoding into the input array
                    inputArray[i * 3 + eyes.sensedType] = eyes.sensedProximity / eyes.maxRange; // normalize to [0,1]
                }
            }

            // get action from brain
            var actionIndex = this.brain.Forward(new ConvNetSharp.Volume.Double.Volume(inputArray, new Shape(inputArray.Length)));
            var action = this.actions[actionIndex];
            this.actionIndex = actionIndex; //back this up

            // demultiplex into behavior variables
            this.rotationSpeed1 = action[0] * 1;
            this.rotationSpeed2 = action[1] * 1;
        }

        public void Backward()
        {
            // in backward pass agent learns.
            // compute reward 
            var proximityReward = 0.0;
            var numEyes = this.eyes.Count;
            for (var i = 0; i < numEyes; i++)
            {
                var e = this.eyes[i];
                // agents dont like to see walls, especially up close
                proximityReward += e.sensedType == 0 ? e.sensedProximity / e.maxRange : 1.0;
            }
            proximityReward = proximityReward / numEyes;
            proximityReward = Math.Min(1.0, proximityReward * 2);

            // agents like to go straight forward
            var forwardReward = 0.0;
            if (this.actionIndex == 0 && proximityReward > 0.75) forwardReward = 0.1 * proximityReward;

            // agents like to eat good things
            var digestionReward = this.digestionSignal;
            this.digestionSignal = 0.0;

            var reward = proximityReward + forwardReward + digestionReward;

            // pass to brain for learning
            this.brain.Backward(reward);
        }
        #endregion
    }
}