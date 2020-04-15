//: Нейросеть распознающая все цифры

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

    // Считываем матрицы обучающей выборки
    getDataFromTextFile(data_x, "./resources/TeachChoose.txt");
    for(int i = 0; i < 10; i++){
        data_y[0][i] = Tensor<double >(1, 10, 1);
        data_y[0][i].Fill(0);
        data_y[0][i][0][i][0] = 1;
    }

    Classifier.learnModel(data_x, data_y, 1, 20, G, rmsErrorD, metrixes);
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