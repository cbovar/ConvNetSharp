using System.Collections.Generic;

namespace ConvNetSharp
{
    public interface IVolume : IEnumerable<double>
    {
        void Add(int x, int y, int d, double v);

        void AddFrom(IVolume volume);

        void AddFromScaled(IVolume volume, double a);

        void AddGradient(int x, int y, int d, double v);

        void AddGradientFrom(IVolume volume);

        IVolume Clone();

        IVolume CloneAndZero();

        double Get(int x, int y, int d);

        double GetGradient(int x, int y, int d);

        void Set(int x, int y, int d, double v);

        void SetConst(double c);

        void SetGradient(int x, int y, int d, double v);

        double GetWeight(int i);

        void SetWeight(int i, double v);

        double GetWeightGradient(int i);

        void SetWeightGradient(int i, double v);

        int Width { get; }

        int Height { get; }

        int Depth { get; }

        int Length { get; }

        void ZeroGradients();
    }
}