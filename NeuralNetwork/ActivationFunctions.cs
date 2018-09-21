using System;

namespace NeuralNetwork {
    // тип функции активации
    public enum ActivationType {
        sigmoid,
        tanh,
        relu,
        nochange
    };

    public delegate double ActivationFunction(double x);

    class ActivationFunctions {
        // Сигмоида. Область значений: (0, 1)
        public static double Sigmoid(double x) {
            return 1.0 / (1 + Math.Exp(-x));
        }

        // производная сигмоиды
        public static double SigmoidDerivative(double x) {
            double f = Sigmoid(x);
            return f * (1 - f);
        }

        // Гиперболический тангенс. Область значений: (-1, 1)
        public static double Tangent(double x) {
            return Math.Tanh(x);
        }

        // производная гиперболического тангенса
        public static double TangentDerivative(double x) {
            double f = Tangent(x);

            return 1 - f * f;
        }
        
        // Выпрямитель. Область значений: [0, +inf)
        public static double ReLU(double x) {
            if (x < 0)
                return 0;

            return x;
        }

        // производная выпрямителя
        public static double ReLUDerivative(double x) {
            if (x < 0)
                return 0;

            return 1;
        }

        // Линейная функция. Область значений: (-inf, +inf)
        public static double NoChange(double x) {
            return x;
        }

        // производная линейной функции
        public static double NoChangeDerivative(double x) {
            return 1;
        }
        
        // Активация значения x функцией типа type
        public static ActivationFunction GetFunction(ActivationType type) {
            switch (type) {
                case ActivationType.sigmoid:
                    return Sigmoid;

                case ActivationType.tanh:
                    return Tangent;

                case ActivationType.relu:
                    return ReLU;

                case ActivationType.nochange:
                    return NoChange;
            }

            throw new Exception("ActivationFunctions: uncased type!");
        }

        // получение производной функции активации с типом type
        public static ActivationFunction GetDerivative(ActivationType type) {
            switch (type) {
                case ActivationType.sigmoid:
                    return SigmoidDerivative;

                case ActivationType.tanh:
                    return TangentDerivative;

                case ActivationType.relu:
                    return ReLUDerivative;

                case ActivationType.nochange:
                    return NoChangeDerivative;
            }

            throw new Exception("ActivationFunctions: uncased type!");
        }
    }
}
