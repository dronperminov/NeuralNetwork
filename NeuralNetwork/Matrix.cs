﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    public class Matrix {
        public readonly int n; // число строк
        public readonly int m; // число столбцов

        double[][] values; // значения матрицы

        public Matrix(int n, int m) {
            if (n < 1 || m < 1)
                throw new Exception("Matrix: n and m must be greater than zero");

            this.n = n;
            this.m = m;

            values = new double[n][];

            for (int i = 0; i < n; i++)
                values[i] = new double[m];
        }

        // заполнение матрицы случайными числами из [a, b)
        public void SetRandom(double a = -0.5, double b = 0.5) {
            Random random = new Random(DateTime.Now.Millisecond);
            double width = b - a;

            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    values[i][j] = a + random.NextDouble() * width;
        }

        public double this[int i, int j] {
            get { return values[i][j]; }
            set { values[i][j] = value; }
        }

        // умножение транспонированной матрицы m на вектор v
        public static Vector operator ^(Matrix m, Vector v) {
            if (m.n != v.GetLength())
                throw new Exception("Matrix: try multiply transpose matrix with rows1 != rows2");

            Vector result = new Vector(m.m);

            for (int i = 0; i < m.m; i++) {
                double sum = 0;

                for (int j = 0; j < m.n; j++)
                    sum += m.values[j][i] * v[j];

                result[i] = sum;
            }

            return result;
        }

        // умножение матрицы m на вектор v
        public static Vector operator *(Matrix m, Vector v) {
            if (m.m != v.GetLength())
                throw new Exception("Matrix: try multiply matrix to vector with different sizes");

            Vector result = new Vector(m.n);

            for (int i = 0; i < m.n; i++) {
                double sum = 0;

                for (int j = 0; j < m.m; j++)
                    sum += m.values[i][j] * v[j];

                result[i] = sum;
            }

            return result;
        }

        // вывод матрицы в консоль
        public void Print() {
            Console.WriteLine("Matrix: [{0} x {1}]", n, m);

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++)
                    Console.Write("{0}  ", values[i][j]);

                Console.WriteLine();
            }

            Console.WriteLine();
        }
    }
}