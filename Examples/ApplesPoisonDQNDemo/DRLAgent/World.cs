using ConvNetSharp.ReinforcementLearning.Deep;
using ConvnetSharpOLD;
using System;
using System.Collections.Generic;

namespace DeepQLearning.DRLAgent
{
    // World object contains many agents and walls and food and stuff
    [Serializable]
    public class World
    {
        #region Member Fields
        private Util _util;

        private int _worldWidth, _worldHeight;
        private int _clock;

        public List<Wall> _walls;
        public List<Item> _items;
        public List<Agent> agents;

        private List<Intersect> _collisionPoints;

        private const double _digestionSignalApple = 5.0;
        private const double _digestionSignalPoison = -6.0;
        #endregion

        #region Member Properties
        public List<Wall> Walls
        {
            get { return _walls; }
        }

        public List<Item> Items
        {
            get { return _items; }
        }
        #endregion

        #region Constructor
        public World(DeepQLearner brain, int canvasWidth, int canvasHeight)
        {
            agents = new List<Agent>();
            _worldWidth = canvasWidth;
            _worldHeight = canvasHeight;

            _util = new Util();
            _clock = 0;

            // set up walls in the world
            _walls = new List<Wall>();
            var wallPadding = 10;

            AddBox(_walls, wallPadding, wallPadding, _worldWidth - wallPadding * 2, _worldHeight - wallPadding * 2);
            AddBox(_walls, 100, 100, 200, 300); // inner walls

            _walls.RemoveAt(_walls.Count - 1);
            AddBox(_walls, 400, 100, 200, 300);
            _walls.RemoveAt(_walls.Count - 1);

            // set up food and poison
            _items = new List<Item>();
            for (var k = 0; k < 30; k++)
            {
                var x = _util.Randf(20, _worldWidth - 20);
                var y = _util.Randf(20, _worldHeight - 20);
                var t = _util.RandInt(1, 3); // food or poison (1 and 2)
                var it = new Item(x, y, t);
                _items.Add(it);
            }

            // instantiate agent
            agents = new List<Agent>();
            agents.Add(new Agent(brain));
        }
        #endregion

        #region Public Functions
        public void Tick()
        {
            // tick the environment
            _clock++;

            // fix input to all agents based on environment process eyes
            _collisionPoints = new List<Intersect>();
            for (int i = 0, n = agents.Count; i < n; i++)
            {
                var agent = agents[i];
                for (int ei = 0, ne = agent.eyes.Count; ei < ne; ei++)
                {
                    var e = agent.eyes[ei];
                    // we have a line from p to p->eyep
                    var eyep = new Vector(agent.p.x + e.maxRange * Math.Sin(agent.angle + e.angle), agent.p.y + e.maxRange * Math.Cos(agent.angle + e.angle));
                    var res = StuffCollision(agent.p, eyep, true, true);

                    if (res.intersect)
                    {
                        // eye collided with wall
                        e.sensedProximity = res.up.dist_from(agent.p);
                        e.sensedType = res.type;
                    }
                    else
                    {
                        e.sensedProximity = e.maxRange;
                        e.sensedType = -1;
                    }
                }
            }

            // let the agents behave in the world based on their input
            for (int i = 0, n = agents.Count; i < n; i++)
            {
                agents[i].Forward();
            }

            // apply outputs of agents on evironment
            for (int i = 0, n = agents.Count; i < n; i++)
            {
                var agent = agents[i];
                agent.op = agent.p; // back up old position
                agent.oangle = agent.angle; // and angle

                // steer the agent according to outputs of wheel velocities
                var vec = new Vector(0, agent.rad / 2.0);
                vec = vec.Rotate(agent.angle + Math.PI / 2);
                var w1p = agent.p.Add(vec); // positions of wheel 1 and 2
                var w2p = agent.p.Sub(vec);
                var vv = agent.p.Sub(w2p);
                vv = vv.Rotate(-agent.rotationSpeed1);
                var vv2 = agent.p.Sub(w1p);
                vv2 = vv2.Rotate(agent.rotationSpeed2);
                var np = w2p.Add(vv);
                np.Scale(0.5);
                var np2 = w1p.Add(vv2);
                np2.Scale(0.5);
                agent.p = np.Add(np2);

                agent.angle -= agent.rotationSpeed1;
                if (agent.angle < 0) agent.angle += 2 * Math.PI;
                agent.angle += agent.rotationSpeed2;
                if (agent.angle > 2 * Math.PI) agent.angle -= 2 * Math.PI;

                // agent is trying to move from p to op. Check walls
                var res = StuffCollision(agent.op, agent.p, true, false);
                if (res.intersect)
                {
                    // wall collision! reset position
                    agent.p = agent.op;
                }

                // handle boundary conditions
                if (agent.p.x < 0) agent.p.x = 0;
                if (agent.p.x > _worldWidth) agent.p.x = _worldWidth;
                if (agent.p.y < 0) agent.p.y = 0;
                if (agent.p.y > _worldHeight) agent.p.y = _worldHeight;
            }

            // tick all items
            var updateItems = false;
            for (int i = 0, n = _items.Count; i < n; i++)
            {
                var it = _items[i];
                it.age += 1;

                // see if some agent gets lunch
                for (int j = 0, m = agents.Count; j < m; j++)
                {
                    var a = agents[j];
                    var d = a.p.dist_from(it.position);
                    if (d < it.rad + a.rad)
                    {

                        // wait lets just make sure that this isn't through a wall
                        var recheck = StuffCollision(a.p, it.position, true, false);
                        if (!recheck.intersect)
                        {
                            // ding! nom nom nom
                            if (it.type == 1) a.digestionSignal += _digestionSignalApple; // mmm delicious apple
                            if (it.type == 2) a.digestionSignal += _digestionSignalPoison; // ewww poison
                            it.cleanup_ = true;
                            updateItems = true;
                            break; // break out of loop, item was consumed
                        }
                    }
                }

                if (it.age > 5000 && _clock % 100 == 0 && _util.Randf(0, 1) < 0.1)
                {
                    it.cleanup_ = true; // replace this one, has been around too long
                    updateItems = true;
                }
            }
            if (updateItems)
            {
                var nt = new List<Item>();
                for (int i = 0, n = _items.Count; i < n; i++)
                {
                    var it = _items[i];
                    if (!it.cleanup_) nt.Add(it);
                }
                _items = nt; // swap
            }
            if (_items.Count < 30 && _clock % 10 == 0 && _util.Randf(0, 1) < 0.25)
            {
                var newitx = _util.Randf(20, _worldWidth - 20);
                var newity = _util.Randf(20, _worldHeight - 20);
                var newitt = _util.RandInt(1, 3); // food or poison (1 and 2)
                var newit = new Item(newitx, newity, newitt);
                _items.Add(newit);
            }

            // agents are given the opportunity to learn based on feedback of their action on environment
            for (int i = 0, n = agents.Count; i < n; i++)
            {
                agents[i].Backward();
            }
        }
        #endregion

        #region Local Functions
        private void AddBox(List<Wall> lst, double x, double y, double w, double h)
        {
            lst.Add(new Wall(new Vector(x, y), new Vector(x + w, y)));
            lst.Add(new Wall(new Vector(x + w, y), new Vector(x + w, y + h)));
            lst.Add(new Wall(new Vector(x + w, y + h), new Vector(x, y + h)));
            lst.Add(new Wall(new Vector(x, y + h), new Vector(x, y)));
        }

        // helper function to get closest colliding walls/items
        private Intersect StuffCollision(Vector p1, Vector p2, bool check_walls, bool check_items)
        {
            Intersect minres = new Intersect() { intersect = false };

            // collide with walls
            if (check_walls)
            {
                for (int i = 0, n = _walls.Count; i < n; i++)
                {
                    var wall = _walls[i];
                    var res = LineIntersection(p1, p2, wall.point1, wall.point2);
                    if (res.intersect)
                    {
                        res.type = 0; // 0 is wall
                        if (!minres.intersect)
                        {
                            minres = res;
                        }
                        else
                        {   // check if its closer
                            if (res.ua < minres.ua)
                            {
                                // if yes replace it
                                minres = res;
                            }
                        }
                    }
                }
            }

            // collide with items
            if (check_items)
            {
                for (int i = 0, n = _items.Count; i < n; i++)
                {
                    var it = _items[i];
                    var res = LinePointIntersection(p1, p2, it.position, it.rad);
                    if (res.intersect)
                    {
                        res.type = it.type; // store type of item
                        if (!minres.intersect) { minres = res; }
                        else
                        {   // check if its closer
                            if (res.ua < minres.ua)
                            {
                                // if yes replace it
                                minres = res;
                            }
                        }
                    }
                }
            }

            return minres;
        }

        // line intersection helper function: does line segment (p1,p2) intersect segment (p3,p4) ?
        private Intersect LineIntersection(Vector p1, Vector p2, Vector p3, Vector p4)
        {
            Intersect result = new Intersect() { intersect = false };

            var denom = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
            if (denom == 0.0) { result.intersect = false; } // parallel lines

            var ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / denom;
            var ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / denom;
            if (ua > 0.0 && ua < 1.0 && ub > 0.0 && ub < 1.0)
            {
                var up = new Vector(p1.x + ua * (p2.x - p1.x), p1.y + ua * (p2.y - p1.y));
                return new Intersect { ua = ua, ub = ub, up = up, intersect = true }; // up is intersection point
            }
            return result;
        }

        private Intersect LinePointIntersection(Vector A, Vector B, Vector C, double rad)
        {

            Intersect result = new Intersect { intersect = false };

            var v = new Vector(B.y - A.y, -(B.x - A.x)); // perpendicular vector
            var d = Math.Abs((B.x - A.x) * (A.y - C.y) - (A.x - C.x) * (B.y - A.y));
            d = d / v.GetMagnitude();
            if (d > rad) { return result; }

            v.Normalize();
            v.Scale(d);
            double ua = 0.0;
            var up = C.Add(v);
            if (Math.Abs(B.x - A.x) > Math.Abs(B.y - A.y))
            {
                ua = (up.x - A.x) / (B.x - A.x);
            }
            else
            {
                ua = (up.y - A.y) / (B.y - A.y);
            }
            if (ua > 0.0 && ua < 1.0)
            {
                result = new Intersect { ua = ua, up = up, intersect = true };
            }
            return result;
        }

        private Boolean AreSimilar(double a, double b, double tolerance)
        {
            // Values are within specified tolerance of each other....
            return Math.Abs(a - b) < tolerance;
        }
        #endregion
    }
}