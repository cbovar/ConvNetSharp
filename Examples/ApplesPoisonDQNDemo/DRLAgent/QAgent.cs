using ConvNetSharp.ReinforcementLearning.Deep;
using System;
using System.Drawing;

namespace DeepQLearning.DRLAgent
{
    [Serializable]
    public struct Intersect
    {
        public double ua;
        public double ub;
        public Vector up;
        public int type;
        public bool intersect;
    };

    [Serializable]
    public class QAgent
    {
        #region Member Fields
        private int _simulationSpeed = 1;
        private World _world;

        [NonSerialized]
        private Pen _greenPen = new Pen(Color.LightGreen, 2);
        [NonSerialized]
        private Pen _redPen = new Pen(Color.Red, 2);
        [NonSerialized]
        private Pen _greenPen2 = new Pen(Color.LightGreen, 1);
        [NonSerialized]
        private Pen _redPen2 = new Pen(Color.Red, 1);
        [NonSerialized]
        private Pen _bluePen = new Pen(Color.Blue, 2);
        [NonSerialized]
        private Pen _blackPen = new Pen(Color.Black);
        #endregion

        #region Member Properties
        public int SimulationSpeed
        {
            get { return _simulationSpeed; }
            set { _simulationSpeed = value; }
        }
        #endregion

        #region Constructor
        public QAgent(DeepQLearner brain, int canvas_W, int canvas_H)
        {
            _world = new World(brain, canvas_W, canvas_H);
        }
        #endregion

        #region Public Functions
        public void Reinitialize()
        {
            _greenPen = new Pen(Color.LightGreen, 2);
            _redPen = new Pen(Color.Red, 2);
            _greenPen2 = new Pen(Color.LightGreen, 1);
            _redPen2 = new Pen(Color.Red, 1);
            _bluePen = new Pen(Color.Blue, 2);
            _blackPen = new Pen(Color.Black);

            _simulationSpeed = 1;
            _world.agents[0].brain.IsLearning = false;

            _world.agents[0].op.x = 500;
            _world.agents[0].op.y = 500;
        }

        public void Tick()
        {
            _world.Tick();
        }

        // Draw everything and return stats
        public string DrawWorld(Graphics graphics)
        {
            var agents = _world.agents;

            // draw walls in environment
            for (int i = 0, n = _world.Walls.Count; i < n; i++)
            {
                var q = _world.Walls[i];
                DrawLine(graphics, q.point1, q.point2, _blackPen);
            }

            // draw agents
            for (int i = 0, n = agents.Count; i < n; i++)
            {
                // draw agent's body
                var a = agents[i];
                DrawArc(graphics, a.op, (int)a.rad, 0, (float)(Math.PI * 2), _blackPen);

                // draw agent's sight
                for (int ei = 0, ne = a.eyes.Count; ei < ne; ei++)
                {
                    var e = a.eyes[ei];
                    var sr = e.sensedProximity;
                    Pen pen;

                    if (e.sensedType == 1) pen = _redPen2;           // apples
                    else if (e.sensedType == 2) pen = _greenPen2;    // poison
                    else pen = _blackPen;                            // wall

                    //var new_x = a.op.x + sr * Math.Sin(radToDegree((float)a.oangle) + radToDegree((float)e.angle));
                    //var new_y = a.op.y + sr * Math.Cos(radToDegree((float)a.oangle) + radToDegree((float)e.angle));

                    var new_x = a.op.x + sr * Math.Sin(a.oangle + e.angle);
                    var new_y = a.op.y + sr * Math.Cos(a.oangle + e.angle);
                    Vector b = new Vector(new_x, new_y);

                    DrawLine(graphics, a.op, b, pen);
                }
            }

            // draw items
            for (int i = 0, n = _world.Items.Count; i < n; i++)
            {
                Pen pen = _blackPen;
                var it = _world.Items[i];
                if (it.type == 1) pen = _redPen; 
                if (it.type == 2) pen = _greenPen;

                DrawArc(graphics, it.position, (int)it.rad, 0, (float)(Math.PI * 2), pen);
            }

            return _world.agents[0].brain.Age + " " + _world.agents[0].brain.AverageReward;
        }

        public void RunVeryFast()
        {
            _simulationSpeed = 3;
        }

        public void RunFast()
        {
            _simulationSpeed = 2;
        }

        public void RunNormal()
        {
            _simulationSpeed = 1;
        }

        public void RunSlow()
        {
            _simulationSpeed = 0;
        }

        public void StartLearn()
        {
            _world.agents[0].brain.IsLearning = true;
        }

        public void StopLearn()
        {
            _world.agents[0].brain.IsLearning = false;            
        }
        #endregion

        #region Local Functions
        private void DrawCircle(Graphics graphics, Vector center, int radius, Pen pen)
        {
            var rect = new Rectangle((int)center.x - radius, (int)center.y - radius, radius * 2, radius * 2);
            graphics.DrawEllipse(pen, rect);
        }

        private void DrawArc(Graphics graphics, Vector center, int radius, float startAngle, float sweepAngle, Pen pen)
        {
            var rect = new Rectangle((int)center.x - radius, (int)center.y - radius, radius * 2, radius * 2);
            graphics.DrawArc(pen, rect, RadToDegree(startAngle), RadToDegree(sweepAngle));
        }

        private void DrawLine(Graphics graphics, Vector pointA, Vector pointB, Pen pen)
        {
            Point[] points =
            {
                new Point((int)pointA.x, (int)pointA.y),
                new Point((int)pointB.x, (int)pointB.y)
            };

            graphics.DrawLines(pen, points);
        }

        private float RadToDegree(float rad)
        {
            return (float)(rad * 180 / Math.PI);
        }
        #endregion
    }
}
