using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace ConvNetSharp.Volume.Tests
{
    public static class AssertNumber
    {
        public static void AreEqual<T>(double expected, T actual, double delta = 0)
        {
            var value = (double)Convert.ChangeType(actual, typeof(double));
            Assert.AreEqual(expected, value, delta);
        }

        public static void AreEqual<T>(T expected, T actual, double delta = 0)
        {
            var expval = (double)Convert.ChangeType(expected, typeof(double));
            var value = (double)Convert.ChangeType(actual, typeof(double));
            Assert.AreEqual(expval, value, delta);
        }

        public static void AreSequenceEqual<T>(IEnumerable<T> expected, IEnumerable<T> actual, double delta = 0)
        {
            using (var enumerator1 = expected.GetEnumerator())
            {
                using (var enumerator2 = actual.GetEnumerator())
                {
                    while (enumerator1.MoveNext())
                    {
                        enumerator2.MoveNext();

                        AreEqual(enumerator1.Current, enumerator2.Current, delta);
                    }
                }
            }
        }
    }
}