﻿//: Нейросеть распознающая все цифры

#include "Model.h"
#include <vector>
#include <csv.h>
#include <iostream>

// Макрос режима работы программы (с обучением или без)
#define Teach

// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;
using namespace ANN;

int main()
{
    // Создание функтора
    Sigm<double > F(1);
    Relu<double > F1(1);

    // Создание производной функтора
    SigmD<double > FD(1);
    ReluD<double > FD1(1);

    RMS_error<double> rmsError;
    Accuracy<double> accuracy1;
    RMS_errorD<double> rmsErrorD;
    vector<Metr<double>*> metrixes;
    metrixes.push_back(&accuracy1);
    metrixes.push_back(&rmsError);

    glorot_uniform<double> I1(15, 50);
    glorot_uniform<double> I2(50, 10);

    // Создание весов нейросети
    DenceLayer<double> layer1(50, 15, F1, FD1, I1);
    DenceLayer<double> layer2(10, 50, F, FD, I2);

    // Массив выходов первого слоя сети
    Model<double> Classifier = Model<double >();

    Classifier.add(&layer1);
    Classifier.add(&layer2);

    double summ; // Переменная суммы

#ifdef Teach

    // Последовательность цифр, тасуемая для получения равномерной рандомизации
    SGD<double> G(0.09);

    // Создание обучающей выборки
    Matrix<Tensor<double>> data_x(1, 10);
    Matrix<Tensor<double>> data_y(1, 10);
    for(int i = 0; i < 10; i++){
        data_x[0][i] = Tensor<double>(1, 15, 1);
        data_y[0][i] = Tensor<double>(1, 1, 1);
    }

    // Считываем матрицы обучающей выборки
    io::CSVReader<16> in("./resources/training_nums.csv");
    for(int i = 0; i < 10; i++){
        in.read_row(data_y[0][i][0][0][0], data_x[0][i][0][0][0],
                    data_x[0][i][0][0][1], data_x[0][i][0][0][2],
                    data_x[0][i][0][0][3], data_x[0][i][0][0][4],
                    data_x[0][i][0][0][5], data_x[0][i][0][0][6],
                    data_x[0][i][0][0][7], data_x[0][i][0][0][8],
                    data_x[0][i][0][0][9], data_x[0][i][0][0][10],
                    data_x[0][i][0][0][11], data_x[0][i][0][0][12],
                    data_x[0][i][0][0][13], data_x[0][i][0][0][14]);

        auto tmp = Tensor<double >(1,10,1);
        tmp.Fill(0);
        tmp[0][0][int(data_y[0][i][0][0][0])] = 1;
        data_y[0][i] = tmp;
    }

    Classifier.learnModel(data_x, data_y, 1, 600, G, rmsErrorD, metrixes);
    Classifier.saveWeight();
#else
    // Считывание весов
    Classifier.getWeight();

#endif // Teach

    // Создание тестовой выборки
    Matrix<Tensor<double>> test_x(1, 25);
    Matrix<Tensor<double>> test_y(1, 25);

    // Считывание тестовой выборки из файла
    getDataFromTextFile(test_x, "./resources/Tests.txt");


    return 0;
}