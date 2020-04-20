//: Нейросеть распознающая все цифры

#include "Model.h"
#include <csv.h>
#include <iostream>

// Макрос режима работы программы (с обучением или без)
#define Teach

// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;
using namespace NN;

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
    std::vector<Metr<double>*> metrixes;
    metrixes.push_back(&accuracy1);
    metrixes.push_back(&rmsError);

    glorot_uniform<double> I1(15, 128);
    glorot_uniform<double> I2(128, 10);

    // Создание весов нейросети
    DenceLayer<double> layer1(50, 15, F1, FD1, I1, 0.1);
    DenceLayer<double> layer2(10, 50, F, FD, I2, 0.1);

    // Массив выходов первого слоя сети
    Model<double> Classifier = Model<double >();

    Classifier.add(&layer1);
    Classifier.add(&layer2);

    double summ; // Переменная суммы

#ifdef Teach

    // Последовательность цифр, тасуемая для получения равномерной рандомизации
    SGD<double> G(0.2);

    // Создание обучающей выборки
    Vector<Tensor<double>> data_x(10);
    Vector<Tensor<double>> data_y(10);
    for(int i = 0; i < 10; i++){
        data_x[i] = Tensor<double>(1, 15, 1);
        data_y[i] = Tensor<double>(1, 1, 1);
    }

    // Считываем матрицы обучающей выборки
    io::CSVReader<16> in("./resources/training_nums.csv");
    for(int i = 0; i < 10; i++){
        in.read_row(data_y[i][0][0][0], data_x[i][0][0][0],
                    data_x[i][0][0][1], data_x[i][0][0][2],
                    data_x[i][0][0][3], data_x[i][0][0][4],
                    data_x[i][0][0][5], data_x[i][0][0][6],
                    data_x[i][0][0][7], data_x[i][0][0][8],
                    data_x[i][0][0][9], data_x[i][0][0][10],
                    data_x[i][0][0][11], data_x[i][0][0][12],
                    data_x[i][0][0][13], data_x[i][0][0][14]);

        auto tmp = Tensor<double >(1,10,1);
        tmp.Fill(0);
        tmp[0][0][int(data_y[i][0][0][0])] = 1;
        data_y[i] = tmp;
    }

    Classifier.learnModel(data_x, data_y, 1, 1000, G, rmsErrorD, metrixes);
    Classifier.saveWeight();
#else
    // Считывание весов
    Classifier.getWeight();

#endif // Teach

    // Создание тестовой выборки
    Matrix<Tensor<double>> test_x(1, 90);
    Matrix<Tensor<double>> test_y(1, 90);
    for(int i = 0; i < 99; i++){
        test_x[0][i] = Tensor<double>(1, 15, 1);
        test_y[0][i] = Tensor<double>(1, 1, 1);
    }

    // Считывание тестовой выборки из файла
    io::CSVReader<17> in_test("./resources/test_nums.csv");
    int t;
    for(int i = 0; i < 90; i++){
        in_test.read_row(t,  test_x[0][i][0][0][0],
                    test_x[0][i][0][0][1], test_x[0][i][0][0][2],
                    test_x[0][i][0][0][3], test_x[0][i][0][0][4],
                    test_x[0][i][0][0][5], test_x[0][i][0][0][6],
                    test_x[0][i][0][0][7], test_x[0][i][0][0][8],
                    test_x[0][i][0][0][9], test_x[0][i][0][0][10],
                    test_x[0][i][0][0][11], test_x[0][i][0][0][12],
                    test_x[0][i][0][0][13], test_x[0][i][0][0][14], test_y[0][i][0][0][0]);

        auto tmp = Tensor<double >(1,10,1);
        tmp.Fill(0);
        tmp[0][0][int(test_y[0][i][0][0][0])] = 1;
        test_y[0][i] = tmp;
    }

// Вывод на экран реультатов тестирования сети
    Classifier.evaluate(test_x, test_y, metrixes);

    return 0;
}