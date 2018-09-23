using System;

namespace NeuralNetwork {
    public struct Vector {
        int length; // длина вектора
        public readonly double[] values; // значения вектора

        public Vector(int n) {
            length = n;
            values = new double[length];
        }

        public Vector(double[] array) {
            if (array == null || array.Length == 0)
                throw new Exception("Matrix: array is null or empty");

            length = array.Length;

            values = new double[length];

            for (int i = 0; i < length; i++)
                values[i] = array[i];
        }

        public int GetLength() {
            return length;
        }

        public double this[int i] {
            get { return values[i]; }
            set { values[i] = value; }
        }

        // активация матрицы функцией f
        public Vector Activate(ActivationFunction f) {
            Vector activated = new Vector(length);

            for (int i = 0; i < length; i++)
                activated.values[i] = f(values[i]);

            return activated;
        }

        // получение вектора из производных функции df
        public Vector Derivative(ActivationFunction df) {
            Vector derivative = new Vector(length);

            for(int i = 0; i < length; i++)
                derivative.values[i] = df(values[i]);

            return derivative;
        }

        // вывод вектора в консоль
        public void Print() {
            for (int i = 0; i < length; i++)
                Console.Write("{0}  ", values[i]);

            Console.WriteLine();
        }
    }
}