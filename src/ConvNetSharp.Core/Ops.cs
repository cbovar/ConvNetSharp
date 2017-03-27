using System;
using System.Linq.Expressions;
using System.Reflection;

namespace ConvNetSharp.Core
{
    public class Ops<T> where T : struct, IEquatable<T>
    {
        public static readonly Func<T, T, T> Add;

        public static readonly Func<T, T, T> Multiply;

        public static readonly Func<T, T> Log;

        public static readonly Func<T, T, bool> GreaterThan;

        public static readonly Func<T, T> Negate;

        public static T Zero { get; set; }

        public static T One { get; set; }

        static Ops()
        {
            var firstOperand = Expression.Parameter(typeof(T), "x");
            var secondOperand = Expression.Parameter(typeof(T), "y");

            var addBody = Expression.Add(firstOperand, secondOperand);
            Add = Expression.Lambda<Func<T, T, T>>(addBody, firstOperand, secondOperand).Compile();

            var multBody = Expression.Multiply(firstOperand, secondOperand);
            Multiply = Expression.Lambda<Func<T, T, T>>(multBody, firstOperand, secondOperand).Compile();

            var logMethod = typeof(Math).GetRuntimeMethod("Log", new[] {typeof(T)});
            
            if (typeof(T) == typeof(double))
            {
                Log =
                    Expression.Lambda<Func<T, T>>(
                        Expression.Convert(
                            Expression.Call(null, logMethod, Expression.Convert(firstOperand, typeof(double))),
                            typeof(T)), firstOperand).Compile();
                One = (T)(ValueType)1.0;
            }
            else if (typeof(T) == typeof(float))
            {
                Log =
                    Expression.Lambda<Func<T, T>>(
                        Expression.Convert(
                            Expression.Call(null, logMethod, Expression.Convert(firstOperand, typeof(double))),
                            typeof(T)), firstOperand).Compile();
                One = (T)(ValueType)1.0f;
            }

            GreaterThan =
                Expression.Lambda<Func<T, T, bool>>(Expression.GreaterThan(firstOperand, secondOperand), firstOperand,
                    secondOperand).Compile();

            Negate = Expression.Lambda<Func<T, T>>(Expression.Negate(firstOperand), firstOperand).Compile();

            Zero = default(T);
        }
    }
}