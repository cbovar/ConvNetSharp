using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public class DoubleOpTests : OpTests<double>
    {
        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new Volume.Double.Volume(values, shape);
        }
    }

    [TestClass]
    public class SingleOpTests : OpTests<float>
    {
        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float)i).ToArray();
            return new Volume.Single.Volume(converted, shape);
        }
    }

    [TestClass]
    public abstract class OpTests<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected abstract Volume<T> NewVolume(double[] values, Shape shape);

        [TestMethod]
        public void Reshape()
        {
            var x = new PlaceHolder<T>("x");
            var op = new Reshape<T>(x, new Shape(1, 1, -1, 1));

            using (var session = new Session<T>())
            {
                // [4] -> [1,1,4,1]
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(4)) } });
                Assert.AreEqual(new Shape(1, 1, 4, 1), result.Shape);

                // [8] -> [1,1,8,1]
                result = session.Run(op, new Dictionary<string, Volume<T>>
                {
                    {
                        "x", NewVolume(new[]
                        {
                            1.0, 2.0, 3.0, 4.0,
                            1.0, 2.0, 3.0, 4.0
                        }, Volume.Shape.From(8))
                    }
                });
                Assert.AreEqual(new Shape(1, 1, 8, 1), result.Shape);
            }
        }

        [TestMethod]
        public void ReshapeDerivate()
        {
            var x = new PlaceHolder<T>("x");
            var op = new Reshape<T>(x, new Shape(1, 1, -1, 1));
            var grad = new PlaceHolder<T>("grad");

            using (var session = new Session<T>())
            {
                op.Derivate = grad;
                op.Differentiate();

                var diff = x.Derivate;

                // [4,1,1,1] -> [1,1,4,1]
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(4, 1, 1, 1)) } });

                // [1,1,4,1] -> [4,1,1,1]
                result = session.Run(diff,
                    new Dictionary<string, Volume<T>>
                    {
                        {"x", NewVolume(new[] {1.0, 2.0, 3.0, 4.0}, Volume.Shape.From(4))},
                        {"grad", NewVolume(new[] {1.0, 1.0, 1.0, 1.0}, Volume.Shape.From(1,1,4,1))}
                    });
                Assert.AreEqual(new Shape(4,1,1,1), result.Shape);
            }
        }

        [TestMethod]
        public void Shape()
        {
            var x = new PlaceHolder<T>("x");
            var op = new Shape<T>(x);

            using (var session = new Session<T>())
            {
                // Batch size = 1
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(1, 1, 4, 1)) } });

                AssertNumber.AreEqual(1.0, result.Get(0));
                AssertNumber.AreEqual(1.0, result.Get(1));
                AssertNumber.AreEqual(4.0, result.Get(2));
                AssertNumber.AreEqual(1.0, result.Get(3));

                // Batch size = 2
                result = session.Run(op, new Dictionary<string, Volume<T>>
                {
                    {
                        "x", NewVolume(new[]
                        {
                            1.0, 2.0, 3.0, 4.0,
                            1.0, 2.0, 3.0, 4.0
                        }, Volume.Shape.From(1, 1, 4, 2))
                    }
                });

                AssertNumber.AreEqual(1.0, result.Get(0));
                AssertNumber.AreEqual(1.0, result.Get(1));
                AssertNumber.AreEqual(4.0, result.Get(2));
                AssertNumber.AreEqual(2.0, result.Get(3));
            }
        }
    }
}