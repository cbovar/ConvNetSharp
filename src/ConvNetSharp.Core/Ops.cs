using System;
using System.Linq.Expressions;
using System.Reflection;

namespace ConvNetSharp.Core
{
    public class Ops<T> where T : struct, IEquatable<T>
    {
        public static readonly Func<T, T, T> Add;

        public static readonly Func<T, T, T> Multiply;

        public static readonly Func<T, T, T> Divide;

        public static readonly Func<T, T> Log;

        public static readonly Func<T, T, bool> GreaterThan;

        public static readonly Func<T, T> Negate;

        public static readonly Func<int, T> Cast;

        public static readonly T Zero;

        public static readonly T One;

        public static readonly T Epsilon;

        public static readonly Func<T, bool> IsInvalid;

        static Ops()
        {
            var firstOperand = Expression.Parameter(typeof(T), "x");
            var secondOperand = Expression.Parameter(typeof(T), "y");

            var addBody = Expression.Add(firstOperand, secondOperand);
            Add = Expression.Lambda<Func<T, T, T>>(addBody, firstOperand, secondOperand).Compile();

            var multBody = Expression.Multiply(firstOperand, secondOperand);
            Multiply = Expression.Lambda<Func<T, T, T>>(multBody, firstOperand, secondOperand).Compile();

            var divBody = Expression.Divide(firstOperand, secondOperand);
            Divide = Expression.Lambda<Func<T, T, T>>(divBody, firstOperand, secondOperand).Compile();

            var intOperand = Expression.Parameter(typeof(int), "x");
            var castBody = Expression.Convert(intOperand, typeof(T));
            Cast = Expression.Lambda<Func<int, T>>(castBody, intOperand).Compile();

            //Log exists only as Math.Log(double x) so always to cast to and from double
            var logMethod = typeof(Math).GetRuntimeMethod("Log", new[] {typeof(T)});
            Log = Expression.Lambda<Func<T, T>>(
                Expression.Convert(
                    Expression.Call(null, logMethod, Expression.Convert(firstOperand, typeof(double))),
                    typeof(T)), firstOperand).Compile();

            GreaterThan =
                Expression.Lambda<Func<T, T, bool>>(Expression.GreaterThan(firstOperand, secondOperand), firstOperand,
                    secondOperand).Compile();

            Negate = Expression.Lambda<Func<T, T>>(Expression.Negate(firstOperand), firstOperand).Compile();

            Zero = default(T);

            var nanMethod = typeof(T).GetMethod("IsNaN", new[] {typeof(T)});
            var infMethod = typeof(T).GetMethod("IsInfinity", new[] {typeof(T)});
            IsInvalid = Expression.Lambda<Func<T, bool>>(
                Expression.OrElse(
                    Expression.Call(nanMethod, firstOperand),
                    Expression.Call(infMethod, firstOperand)), firstOperand).Compile();

            if (typeof(T) == typeof(double))
            {
                One = (T) (ValueType) 1.0;
                Epsilon = (T) (ValueType) double.Epsilon;
            }
            else if (typeof(T) == typeof(float))
            {
                One = (T) (ValueType) 1.0f;
                Epsilon = (T) (ValueType) float.Epsilon;
            }
        }
    }
}