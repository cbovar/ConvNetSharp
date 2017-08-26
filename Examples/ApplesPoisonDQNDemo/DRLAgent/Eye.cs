using System;

namespace DeepQLearning.DRLAgent
{
    // Eye sensor has a maximum range and senses walls
    [Serializable]
    public class Eye
    {
        public double angle;
        public double maxRange;
        public double sensedProximity;
        public int sensedType;

        public Eye(double angle)
        {
            this.angle = angle; // angle of the eye relative to the agent
            this.maxRange = 85; // maximum proximity range
            this.sensedProximity = 85; // proximity of what the eye is seeing. will be set in world.Tick()
            this.sensedType = -1; // what type of object does the eye see?
        }
    }
}
