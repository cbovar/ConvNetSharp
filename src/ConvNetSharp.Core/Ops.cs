using System;
using System.Linq.Expressions;
using System.Reflection;

namespace ConvNetSharp.Core
{
    public class Ops<T> where T : struct, IEquatable<T>
    {
        public static readonly Func<T, T, T> Add;

        public static readonly Func<T, T, T> Subtract;

        public static readonly Func<T, T, T> Multiply;

        public static readonly Func<T, T, T> Divide;

        public static readonly Func<T, T> Log;

        public static readonly Func<T, T, T> Pow;

        public static readonly Func<T, T, bool> GreaterThan;

        public static readonly Func<T, T> Negate;

        public static readonly Func<T, T> Sqrt;

        public static readonly Func<double, T> Cast;

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

            var subtractBody = Expression.Subtract(firstOperand, secondOperand);
            Subtract = Expression.Lambda<Func<T, T, T>>(subtractBody, firstOperand, secondOperand).Compile();

            var multBody = Expression.Multiply(firstOperand, secondOperand);
            Multiply = Expression.Lambda<Func<T, T, T>>(multBody, firstOperand, secondOperand).Compile();

            var divBody = Expression.Divide(firstOperand, secondOperand);
            Divide = Expression.Lambda<Func<T, T, T>>(divBody, firstOperand, secondOperand).Compile();

            //Log only exists as Math.Log(double x) so always to cast to and from double
            var logMethod = typeof(Math).GetRuntimeMethod("Log", new[] { typeof(T) });
            Log = Expression.Lambda<Func<T, T>>(
                Expression.Convert(
                    Expression.Call(null, logMethod, Expression.Convert(firstOperand, typeof(double))),
                    typeof(T)), firstOperand).Compile();

            GreaterThan =
                Expression.Lambda<Func<T, T, bool>>(Expression.GreaterThan(firstOperand, secondOperand), firstOperand,
                    secondOperand).Compile();

            Negate = Expression.Lambda<Func<T, T>>(Expression.Negate(firstOperand), firstOperand).Compile();

            Zero = default;

            //Pow only exists as Math.Pow(double x, double y) so always to cast to and from double
            var powMethod = typeof(Math).GetRuntimeMethod("Pow", new[] { typeof(T), typeof(T) });
            Pow = Expression.Lambda<Func<T, T, T>>(
                Expression.Convert(Expression.Call(null, powMethod, Expression.Convert(firstOperand, typeof(double)), Expression.Convert(secondOperand, typeof(double))),
                    typeof(T)), firstOperand, secondOperand).Compile();

            //Sqrt only exists as Math.Sqrt(double d) so always to cast to and from double
            var sqrtMethod = typeof(Math).GetRuntimeMethod("Sqrt", new[] { typeof(T) });
            Sqrt = Expression.Lambda<Func<T, T>>(
                Expression.Convert(
                    Expression.Call(null, sqrtMethod, Expression.Convert(firstOperand, typeof(double))),
                    typeof(T)), firstOperand).Compile();

            var nanMethod = typeof(T).GetRuntimeMethod("IsNaN", new[] { typeof(T) });
            var infMethod = typeof(T).GetRuntimeMethod("IsInfinity", new[] { typeof(T) });
            IsInvalid = Expression.Lambda<Func<T, bool>>(
                Expression.OrElse(
                    Expression.Call(nanMethod, firstOperand),
                    Expression.Call(infMethod, firstOperand)), firstOperand).Compile();

            if (typeof(T) == typeof(double))
            {
                One = (T)(ValueType)1.0;
                Epsilon = (T)(ValueType)double.Epsilon;

                Expression<Func<T, T>> identity = e => e;
                Cast = Expression.Lambda<Func<double, T>>(identity.Body, identity.Parameters[0]).Compile();
            }
            else if (typeof(T) == typeof(float))
            {
                One = (T)(ValueType)1.0f;
                Epsilon = (T)(ValueType)float.Epsilon;

                var doubleOperand = Expression.Parameter(typeof(double), "x");
                var castBody = Expression.Convert(doubleOperand, typeof(T));
                Cast = Expression.Lambda<Func<double, T>>(castBody, doubleOperand).Compile();
            }
        }
    }
}