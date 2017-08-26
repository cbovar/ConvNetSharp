// A 2D vector utility
using System;

[Serializable]
public class Vector
{
    public double x, y;

    public Vector(double x, double y)
    {
        this.x = x;
        this.y = y;
    }

    // utilities
    public double dist_from(Vector vec) { return Math.Sqrt(Math.Pow(this.x - vec.x, 2) + Math.Pow(this.y - vec.y, 2)); }
    public double GetMagnitude() { return Math.Sqrt(Math.Pow(this.x, 2) + Math.Pow(this.y, 2)); }

    // new vector returning operations
    public Vector Add(Vector vec) { return new Vector(this.x + vec.x, this.y + vec.y); }
    public Vector Sub(Vector vec) { return new Vector(this.x - vec.x, this.y - vec.y); }
    public Vector Rotate(double angle)
    {  // CLOCKWISE
        return new Vector(this.x * Math.Cos(angle) + this.y * Math.Sin(angle),
                       -this.x * Math.Sin(angle) + this.y * Math.Cos(angle));
    }

    // in place operations
    public void Scale(double s) { this.x *= s; this.y *= s; }
    public void Normalize() { var d = this.GetMagnitude(); this.Scale(1.0 / d); }
}