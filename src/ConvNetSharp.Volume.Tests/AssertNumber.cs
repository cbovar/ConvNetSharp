using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.Tests
{
    public static class AssertNumber
    {
        public static void AreEqual<T>(double expected, T actual, double delta = 0)
        {
            var value = (double) Convert.ChangeType(actual, typeof(double));
            Assert.AreEqual(expected, value, delta);
        }

        public static void AreEqual<T>(T expected, T actual, double delta = 0)
        {
            var expval = (double) Convert.ChangeType(expected, typeof(double));
            var value = (double) Convert.ChangeType(actual, typeof(double));
            Assert.AreEqual(expval, value, delta);
        }
    }
}