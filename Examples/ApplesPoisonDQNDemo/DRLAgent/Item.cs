using System;

namespace DeepQLearning.DRLAgent
{
    // item is circle thing on the floor that agent can interact with (see or eat, etc)
    [Serializable]
    public class Item
    {
        public Vector position;
        public int type;
        public double rad;
        public int age;
        public bool cleanup_;

        public Item(double x, double y, int type)
        {
            this.position = new Vector(x, y); // position
            this.type = type;
            this.rad = 10; // default radius
            this.age = 0;
            this.cleanup_ = false;
        }
    }
}
