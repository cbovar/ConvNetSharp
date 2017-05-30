using System;
using System.Diagnostics;

namespace ConvNetSharp.Volume
{
    [DebuggerDisplay("Volume {Shape.PrettyPrint()}")]
    public abstract class Volume<T> : IDisposable
        where T : struct, IEquatable<T>, IFormattable
    {
        protected Volume()
        {
        }

        protected Volume(VolumeStorage<T> storage)
        {
            this.Storage = storage;
        }

        public VolumeStorage<T> Storage { get; }

        public Shape Shape
        {
            get => this.Storage.Shape;
            set => this.Storage.Shape = value;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Clear()
        {
            this.Storage.Clear();
        }

        public Volume<T> Clone()
        {
            var data = new T[this.Shape.TotalLength];
            Array.Copy(ToArray(), data, data.Length);

            return BuilderInstance<T>.Volume.SameAs(data, this.Shape);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Storage.Dispose();
            }
        }

        public abstract void DoActivation(Volume<T> result, ActivationType type);

        public abstract void DoActivationGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> result, ActivationType type);

        public abstract void DoAdd(Volume<T> other, Volume<T> result);

        public abstract void DoConvolution(Volume<T> filters, int pad, int stride, Volume<T> result);

        public abstract void DoConvolutionGradient(Volume<T> filters, Volume<T> outputGradients,
            Volume<T> inputGradient, Volume<T> filterGradient, int pad, int stride);

        public abstract void DoMultiply(Volume<T> other, Volume<T> result);

        public abstract void DoNegate(Volume<T> result);

        public abstract void DoSoftmax(Volume<T> result);

        public abstract void DoSoftmaxGradient(Volume<T> outputGradient, Volume<T> result);

        public T Get(params int[] coordinates)
        {
            return this.Storage.Get(coordinates);
        }

        public static implicit operator Volume<T>(T t)
        {
            return BuilderInstance<T>.Volume.SameAs(new[] { t }, new Shape(1));
        }

        public static implicit operator Volume<T>(T[] t)
        {
            return BuilderInstance<T>.Volume.SameAs(t, new Shape(t.Length));
        }

        public static implicit operator T(Volume<T> v)
        {
            if (v.Shape.TotalLength == 1)
            {
                return v.Get(0);
            }

            throw new ArgumentException($"Volume should have a Shape [1] to be converter to a {typeof(T)}", nameof(v));
        }

        public Volume<T> ReShape(params int[] dimensions)
        {
            var shape = new Shape(dimensions);
            shape.GuessUnkownDimension(this.Shape.TotalLength);

            return BuilderInstance<T>.Volume.Build(this.Storage, shape);
        }

        public void Set(int[] coordinates, T value)
        {
            this.Storage.Set(coordinates, value);
        }

        public T[] ToArray()
        {
            return this.Storage.ToArray();
        }
    }
}