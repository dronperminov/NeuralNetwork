﻿using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace NeuralNetwork {
    public struct NeuroStructure {
        public int inputs; // число входных нейронов
        public int[] hiddens; // число скрытых нейронов
        public int outputs; // число выходных нейронов

        public ActivationType hiddensFunction; // тип функции активации в скрытых слоях
        public ActivationType outputFunction; // тип функции активации выходного слоя

        public string name; // имя сети
    }

    public struct TrainParameters {
        public double learningRate; // скорость обучения
        public double accuracy; // точность обучения
        public double dropOut; // доля исключаемых нейронов
        public long maxEpochs; // максимальное число эпох обучения
        public bool printTime; // печатать ли время эпохи

        public int autoSavePeriod; // интервал автосохранения сети в файл (0 - не сохранять)
    }

    public class NeuralNetwork {
        NeuroStructure structure; // структура сети

        Matrix[] layers; // слои (матрицы весов)
        Vector[] inputs; // входные значения слоёв
        Vector[] outputs; // выходные значения слоёв

        ActivationFunction hiddensActivation; // функция активации скрытых слоёв
        ActivationFunction hiddensDerivative; // Производная функции активации скрытых слоёв
        ActivationFunction outputActivation; // функция активации выходного слоя
        ActivationFunction outputDerivative; // Производная функции активации выходного слоя

        public delegate void Log(double error, long epoch); // делегат для отслеживания состояния обучения

        public NeuralNetwork(NeuroStructure structure) {
            this.structure = structure;

            Create(); // создаём матрицы весов, входные и выходные сигналы

            // заполняем матрицу маленькими случайными числами
            for (int i = 0; i < layers.Length - 1; i++)
                layers[i].SetRandom(0, Math.Sqrt(2.0 / layers[i].m));
        }

        // создание нейросети из файла, расположенного в path
        public NeuralNetwork(string path) {
            if (!File.Exists(path))
                throw new Exception("NeuralNetwork: fils does not exists");

            StreamReader reader = new StreamReader(path);

            structure.inputs = int.Parse(reader.ReadLine());
            structure.hiddens = new int[int.Parse(reader.ReadLine())];

            for (int i = 0; i < structure.hiddens.Length; i++)
                structure.hiddens[i] = int.Parse(reader.ReadLine());

            structure.outputs = int.Parse(reader.ReadLine());

            structure.hiddensFunction = (ActivationType)int.Parse(reader.ReadLine());
            structure.outputFunction = (ActivationType)int.Parse(reader.ReadLine());

            int index = path.LastIndexOf("\\");

            if (index == -1)
                index = 0;

            structure.name = path.Substring(index).Replace(".", "_");

            Create(); // создаём матрицы весов, входные и выходные сигналы

            for (int layer = 0; layer < layers.Length; layer++) {
                for (int i = 0; i < layers[layer].n; i++) {
                    string row = reader.ReadLine();
                    string[] values = row.Split(' ');

                    for (int j = 0; j < layers[layer].m; j++)
                        layers[layer][i, j] = double.Parse(values[j]);
                }
            }

            reader.Close();
        }

        // создание матриц весов и входных и выходных сигналов
        void Create() {
            if (structure.inputs < 1)
                throw new Exception("Create NeuralNetwork: inputs must be greater than zero");

            if (structure.hiddens.Length == 0)
                throw new Exception("Create NeuralNetwork: hiddens is null or zero");

            for (int i = 0; i < structure.hiddens.Length; i++)
                if (structure.hiddens[i] < 1)
                    throw new Exception("Create NeuralNetwork: hiddens at " + i + " layer must be greater than zero");

            if (structure.outputs < 1)
                throw new Exception("Create NeuralNetwork: outputs must be greater than zero");

            layers = new Matrix[1 + structure.hiddens.Length];
            inputs = new Vector[1 + structure.hiddens.Length];
            outputs = new Vector[1 + structure.hiddens.Length];

            layers[0] = new Matrix(structure.hiddens[0], structure.inputs);

            for (int i = 0; i < structure.hiddens.Length - 1; i++)
                layers[i + 1] = new Matrix(structure.hiddens[i + 1], structure.hiddens[i]);

            layers[layers.Length - 1] = new Matrix(structure.outputs, structure.hiddens[structure.hiddens.Length - 1]);

            hiddensActivation = ActivationFunctions.GetFunction(structure.hiddensFunction);
            hiddensDerivative = ActivationFunctions.GetDerivative(structure.hiddensFunction);

            outputActivation = ActivationFunctions.GetFunction(structure.outputFunction);
            outputDerivative = ActivationFunctions.GetDerivative(structure.outputFunction);
        }

        // получение структуры нейросети
        public NeuroStructure GetStructure() {
            return structure;
        }

        // получение выхода сети для вектора input
        public Vector GetOutput(Vector input) {
            inputs[0] = input;

            // распространяем сигнал от начала к концу
            for (int i = 0; i < layers.Length - 1; i++) {
                outputs[i] = layers[i] * inputs[i];
                inputs[i + 1] = outputs[i].Activate(hiddensActivation);
            }

            int index = layers.Length - 1;
            outputs[index] = layers[index] * inputs[index];

            return outputs[index].Activate(outputActivation); // возвращаем активированный вектор
        }

        // распространение ошибки от последнего к первому слою (f - выход сети, g - требуемый ответ)
        Vector[] PropagateErrors(Vector f, Vector g, ref double error) {
            Vector[] errors = new Vector[layers.Length];

            errors[layers.Length - 1] = new Vector(structure.outputs);

            // считаем ошибку выхода сети
            for (int i = 0; i < structure.outputs; i++) {
                double e = g[i] - f[i]; // компонента ошибки
                errors[layers.Length - 1][i] = e;

                error += e * e; // добавляем квадрат ошибки к общей ошибке
            }

            // распространяем ошибку выше
            for (int i = layers.Length - 2; i >= 0; i--)
                errors[i] = layers[i + 1] ^ errors[i + 1];

            return errors; // возвращаем заполненный массив ошибок
        }

        // получение градиента в каждом слое
        Vector[] GetGradients() {
            Vector[] gradients = new Vector[layers.Length];

            for (int i = 0; i < layers.Length - 1; i++)
                gradients[i] = outputs[i].Derivative(hiddensDerivative);

            gradients[layers.Length - 1] = outputs[layers.Length - 1].Derivative(outputDerivative);

            return gradients; // возвращаем вычисленные градиенты
        }

        void DropOut(double p) {
            Random random = new Random(DateTime.Now.Millisecond);

            for (int layer = 0; layer < layers.Length - 1; layer++) {
                for (int i = 0; i < layers[layer].n; i++) {
                    if (random.NextDouble() < p) {
                        for (int j = 0; j < layers[layer].m; j++) {
                            layers[layer][i, j] = 0;
                        }
                    }
                }
            }
        }

        // обучение сети методом обратного распространения ошибки с моментом
        public void TrainMoment(Vector[] inputData, Vector[] outputData, TrainParameters parameters, double moment, Log log = null) {
            long epoch = 0;
            double error;

            if (parameters.dropOut > 0)
                DropOut(parameters.dropOut);

            Matrix[] dw = new Matrix[layers.Length];

            for (int layer = 0; layer < layers.Length; layer++)
                dw[layer] = new Matrix(layers[layer].n, layers[layer].m);

            Stopwatch t = new Stopwatch();

            do {
                error = 0;
                t.Restart();

                for (int index = 0; index < inputData.Length; index++) {
                    Vector f = GetOutput(inputData[index]); // получаем выход сети
                    Vector g = outputData[index];

                    Vector[] errors = PropagateErrors(f, g, ref error); // считаем и распространяем ошибку
                    Vector[] gradients = GetGradients(); // получаем градиенты

                    // изменяем веса в каждом слое
                    for (int layer = 0; layer < layers.Length; layer++) {
                        double exp = Math.Exp(layer / 2);

                        for (int i = 0; i < layers[layer].n; i++) {
                            double delta = parameters.learningRate * errors[layer][i] * gradients[layer][i];

                            for (int j = 0; j < layers[layer].m; j++) {
                                double deltaW = delta * inputs[layer][j];

                                layers[layer][i, j] += deltaW + moment * dw[layer][i, j];
                                dw[layer][i, j] = deltaW;
                            }
                        }
                    }
                }

                error = Math.Sqrt(error);
                epoch++;

                t.Stop();

                if (parameters.printTime)
                    Console.WriteLine("epoch time: {0}", t.ElapsedMilliseconds);

                if (log != null)
                    log(error, epoch);

                if (epoch % parameters.autoSavePeriod == 0)
                    Save(structure.name + "_" + epoch + "_" + error + ".txt");
            } while (error > parameters.accuracy && epoch < parameters.maxEpochs); // повторяем пока не достигнем нужной точности или максимального числа эпох

            if (epoch == parameters.maxEpochs)
                Console.WriteLine("Warning! Max epoch reached!");
        }

        // обучение сети методом обратного распространения ошибки
        public void Train(Vector[] inputData, Vector[] outputData, TrainParameters parameters, Log log = null) {
            long epoch = 0;
            double error;

            if (parameters.dropOut > 0)
                DropOut(parameters.dropOut);

            Stopwatch t = new Stopwatch();

            do {
                error = 0;
                t.Restart();

                for (int index = 0; index < inputData.Length; index++) {
                    Vector f = GetOutput(inputData[index]); // получаем выход сети
                    Vector g = outputData[index];

                    Vector[] errors = PropagateErrors(f, g, ref error); // считаем и распространяем ошибку
                    Vector[] gradients = GetGradients(); // получаем градиенты

                    // изменяем веса в каждом слое
                    for (int layer = 0; layer < layers.Length; layer++) {
                        Parallel.For(0, layers[layer].n, i => {
                            double delta = parameters.learningRate * errors[layer][i] * gradients[layer][i];

                            for (int j = 0; j < layers[layer].m; j++) {                            
                                layers[layer][i, j] += delta * inputs[layer][j];
                            }
                        });
                    }
                }

                error = Math.Sqrt(error);
                epoch++;

                t.Stop();

                if (parameters.printTime)
                    Console.WriteLine("epoch time: {0}", t.ElapsedMilliseconds);

                if (log != null)
                    log(error, epoch);

                if (epoch % parameters.autoSavePeriod == 0)
                    Save(structure.name + "_" + epoch + "_" + error + ".txt");
            } while (error > parameters.accuracy && epoch < parameters.maxEpochs); // повторяем пока не достигнем нужной точности или максимального числа эпох

            if (epoch == parameters.maxEpochs)
                Console.WriteLine("Warning! Max epoch reached!");
        }

        // сохранение сети в файл, находящийся в path
        public void Save(string path) {
            StreamWriter writer = new StreamWriter(path);

            writer.WriteLine(structure.inputs);

            writer.WriteLine(structure.hiddens.Length);

            for (int i = 0; i < structure.hiddens.Length; i++)
                writer.WriteLine(structure.hiddens[i]);

            writer.WriteLine(structure.outputs);

            writer.WriteLine((int)structure.hiddensFunction);
            writer.WriteLine((int)structure.outputFunction);

            for (int layer = 0; layer < layers.Length; layer++) {
                for (int i = 0; i < layers[layer].n; i++) {
                    for (int j = 0; j < layers[layer].m; j++)
                        writer.Write("{0} ", layers[layer][i, j]);

                    writer.WriteLine();
                }
            }

            writer.Close();
        }
    }
}
