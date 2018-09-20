using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    public class Vector {
        int length; // длина вектора
        double[] values; // значения вектора

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

        // активация матрицы функцией типа type
        public Vector Activate(ActivationType type) {
            Vector activated = new Vector(length);

            Parallel.For(0, length, i => {
                activated.values[i] = ActivationFunction.Activate(type, values[i]);
            });

            return activated;
        }

        // получение вектора из производных функции type
        public Vector Derivative(ActivationType type) {
            Vector derivative = new Vector(length);

            Parallel.For(0, length, i => {
                derivative.values[i] = ActivationFunction.Derivative(type, values[i]);
            });

            return derivative;
        }

        // получение квадрата нормы в Евклидовом пространстве
        public double GetNorm() {
            double norm = 0;

            for (int i = 0; i < length; i++)
                norm += values[i] * values[i];

            return norm;
        }

        // вывод вектора в консоль
        public void Print() {
            for (int i = 0; i < length; i++)
                Console.Write("{0}  ", values[i]);

            Console.WriteLine();
        }
    }
}