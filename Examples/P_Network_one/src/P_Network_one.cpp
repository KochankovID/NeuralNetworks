//: Нейросеть распознающая 4
#include "ANN.h"
#include "Data.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

// Макрос режима работы программы (с обучением или без)
//#define Teach

using namespace std;
using namespace NN;

int main() {
    // Создание функции активации
    BinaryClassificator<int> F;
    BinaryClassificatorD<int> FD;

    RMS_error<int> rmsError;
    BinaryAccuracy<int> binaryAccuracy;

    // Создание инициализатора весов
    SimpleInitializator<int> I;

    // Создание cлоя нейрона из одного нейрона
    I_Neyron neyron(1, 15);

    Matrix<double> Metrics(2, 10);

    int summ;
    Matrix<int> y(1, 1); // Переменная выхода сети
    Matrix<int> a(1, 1);
    Vector<int> output(10);
#ifdef Teach
    // Создание обучающей выборки
    Vector<Tensor<int>> data_x(10);
    Vector<Tensor<int>> data_y(10);
    for (int i = 0; i < 10; i++) {
        data_x[i] = data_y[i] = Tensor<int>(1, 15, 1);
    }


    // Считываем матрицы обучающей выборки
    io::CSVReader<16> in("./resources/training_nums.csv");
    for (int i = 0; i < 10; i++) {
        in.read_row(data_y[i][0][0][0], data_x[i][0][0][0],
                    data_x[i][0][0][1], data_x[i][0][0][2],
                    data_x[i][0][0][3], data_x[i][0][0][4],
                    data_x[i][0][0][5], data_x[i][0][0][6],
                    data_x[i][0][0][7], data_x[i][0][0][8],
                    data_x[i][0][0][9], data_x[i][0][0][10],
                    data_x[i][0][0][11], data_x[i][0][0][12],
                    data_x[i][0][0][13], data_x[i][0][0][14]);

        auto tmp = Tensor<int>(1, 10, 1);
        tmp.Fill(0);
        tmp[0][0][int(data_y[i][0][0][0])] = 1;
        data_y[i] = tmp;
    }

    // Обучение сети
    long int epoch = 6; // Количество обучений нейросети

    for (long int i = 0; i < epoch; i++) {
        for (int j = 0; j < 10; j++) {
            summ = neyron.Summator(data_x[j][0]);
            y[0][0] = neyron.FunkActiv(summ, F);
            if (j != 4) {
                // Если текущая цифра не 4, то ожидаемый ответ 0
                a[0][0] = 0;
            } else {
                // Если текущая цифра 4, то ожидаемый ответ 1
                a[0][0] = 1;
            }
            SimpleLearning(a[0][0], y[0][0], neyron, data_x[j][0], 1);
            cout << "||";
            Metrics[0][j] = metric_function(binaryAccuracy, y, a);
            Metrics[1][j] = metric_function(rmsError, y, a);
        }
        cout << " accuracy: ";
        cout << mean(Metrics[0], 10);
        cout << " loss: " << mean(Metrics[1], 10) << endl;
    }

    // Сохраняем веса
    saveWeightsTextFile(neyron, "./resources/Weights.txt");
#else
    // Считывание весов
    getWeightsTextFile(neyron, "./resources/Weights.txt");

#endif // Teach
    // Создание тестовой выборки
    Vector<Tensor<int>> test_x(90);
    Vector<Tensor<int>> test_y(90);
    for (int i = 0; i < 90; i++) {
        test_x[i] = Tensor<int>(1, 15, 1);
        test_y[i] = Tensor<int>(1, 1, 1);
    }

    // Считывание тестовой выборки из файла
    io::CSVReader<17> in_test("./resources/test_nums.csv");
    int t;
    for (int i = 0; i < 90; i++) {
        in_test.read_row(t, test_x[i][0][0][0],
                         test_x[i][0][0][1], test_x[i][0][0][2],
                         test_x[i][0][0][3], test_x[i][0][0][4],
                         test_x[i][0][0][5], test_x[i][0][0][6],
                         test_x[i][0][0][7], test_x[i][0][0][8],
                         test_x[i][0][0][9], test_x[i][0][0][10],
                         test_x[i][0][0][11], test_x[i][0][0][12],
                         test_x[i][0][0][13], test_x[i][0][0][14], test_y[i][0][0][0]);

        auto tmp = Tensor<int>(1, 10, 1);
        tmp.Fill(0);
        tmp[0][0][int(test_y[i][0][0][0])] = 1;
        test_y[i] = tmp;
    }
    // Вывод на экран реультатов тестирования сети на тестовой выборке
    Metrics = Matrix<double>(2, 90);
    cout << endl << "Validation model: " << endl;
    for (int j = 0; j < 90; j++) {
        summ = neyron.Summator(test_x[j][0]);
        y[0][0] = neyron.FunkActiv(summ, F);
        if (getIndexOfMaxElem(test_y[j][0][0], test_y[j][0][0] + 10) != 4) {
            // Если текущая цифра не 4, то ожидаемый ответ 0
            a[0][0] = 0;
        } else {
            // Если текущая цифра 4, то ожидаемый ответ 1
            a[0][0] = 1;
        }
        Metrics[0][j] = metric_function(binaryAccuracy, y, a);
        Metrics[1][j] = metric_function(rmsError, y, a);
    }
    cout << "accuracy: ";
    cout << mean(Metrics[0], 90);
    cout << " loss: " << mean(Metrics[1], 90) << endl;

    return 0;
}
