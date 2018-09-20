using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    // тип функции активации
    public enum ActivationType {
        sigmoid,
        tanh,
        relu,
        nochange
    };

    class ActivationFunction {
        // Сигмоида. Область значений: (0, 1)
        public static double Sigmoid(double x) {
            return 1.0 / (1 + Math.Exp(-x));
        }

        // Гиперболический тангенс. Область значений: (-1, 1)
        public static double Tangent(double x) {
            return Math.Tanh(x);
        }

        // Выпрямитель. Область значений: [0, +inf)
        public static double ReLU(double x) {
            if (x < 0)
                return 0;

            return x;
        }

        // Линейная функция. Область значений: (-inf, +inf)
        public static double NoChange(double x) {
            return x;
        }

        // Активация значения x функцией типа type
        public static double Activate(ActivationType type, double x) {
            switch (type) {
                case ActivationType.sigmoid:
                    return Sigmoid(x);

                case ActivationType.tanh:
                    return Tangent(x);

                case ActivationType.relu:
                    return ReLU(x);

                case ActivationType.nochange:
                    return NoChange(x);
            }

            throw new Exception("ActivationFunctions: uncased type!");
        }

        /*********************************************************/

        // производная сигмоиды
        public static double SigmoidDerivative(double x) {
            double f = Sigmoid(x);
            return f * (1 - f);
        }
        
        // производная гиперболического тангенса
        public static double TangentDerivative(double x) {
            double cosh = Math.Cosh(x);

            return 1 / (cosh * cosh);
        }

        // производная выпрямителя
        public static double ReLUDerivative(double x) {
            if (x < 0)
                return 0;

            return 1;
        }

        // производная линейной функции
        public static double NoChangeDerivative(double x) {
            return 1;
        }

        // получение производной функции активации с типом type
        public static double Derivative(ActivationType type, double x) {
            switch (type) {
                case ActivationType.sigmoid:
                    return SigmoidDerivative(x);

                case ActivationType.tanh:
                    return TangentDerivative(x);

                case ActivationType.relu:
                    return ReLUDerivative(x);

                case ActivationType.nochange:
                    return NoChangeDerivative(x);
            }

            throw new Exception("ActivationFunctions: uncased type!");
        }
    }
}
