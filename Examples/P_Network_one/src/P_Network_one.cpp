//: Нейросеть распознающая 4
#include "ANN.h"
#include "Data.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

// Макрос режима работы программы (с обучением или без)
#define Teach

using namespace std;
using namespace NN;

int main() {
    // Создание функции активации
    I_BinaryClassificator F;

    // Создание метрик
    I_RMS_error rmsError;
    I_BinaryAccuracy binaryAccuracy;

    // Создание инициализатора весов
    I_SimpleInitializator I;

    // Создание нейрона
    I_Neuron neyron(1, 15);

    int summ;
    D_Matrix Metrics(2, 10);
    I_Matrix y(1, 1);
    I_Matrix a(1, 1);
    I_Vector output(10);
#ifdef Teach
    // Создание обучающей выборки
    Vector<I_Tensor> data_x(10);
    Vector<I_Tensor> data_y(10);
    for (int i = 0; i < 10; i++) {
        data_x[i] = data_y[i] = I_Tensor(1, 15, 1);
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
        auto tmp = I_Tensor(1, 10, 1);
        tmp.Fill(0);
        tmp[0][0][int(data_y[i][0][0][0])] = 1;
        data_y[i] = tmp;
    }

    // Обучение сети
    long int epoch = 6; // Количество обучений нейросети
    for (long int i = 0; i < epoch; i++) {
        for (int j = 0; j < 10; j++) { // Проход по обучающей выборке
            summ = neyron.Summator(data_x[j][0]); // Вычисление взвешенной суммы
            y[0][0] = neyron.FunkActiv(summ, F); // Получение результата функции активации
            if (j != 4) {
                // Если текущая цифра не 4, то ожидаемый ответ 0
                a[0][0] = 0;
            } else {
                // Если текущая цифра 4, то ожидаемый ответ 1
                a[0][0] = 1;
            }
            SimpleLearning(a[0][0], y[0][0], neyron, data_x[j][0], 1); // Калибровка весов нейрона
            cout << "||";
            // Вычисление метрик
            Metrics[0][j] = metric_function(binaryAccuracy, y, a);
            Metrics[1][j] = metric_function(rmsError, y, a);
        }
        // Вывод метрик в консоль
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
    Vector<I_Tensor> test_x(90);
    Vector<I_Tensor> test_y(90);
    for (int i = 0; i < 90; i++) {
        test_x[i] = I_Tensor(1, 15, 1);
        test_y[i] = I_Tensor(1, 1, 1);
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

        auto tmp = I_Tensor(1, 10, 1);
        tmp.Fill(0);
        tmp[0][0][int(test_y[i][0][0][0])] = 1;
        test_y[i] = tmp;
    }

    Metrics = D_Matrix(2, 90);

    // Вывод на экран реультатов тестирования сети на тестовой выборке
    cout << endl << "Validation model: " << endl;
    for (int j = 0; j < 90; j++) { // Проход по тестовой выборке
        summ = neyron.Summator(test_x[j][0]); // Вычисление взвешенной суммы
        y[0][0] = neyron.FunkActiv(summ, F); // Получение результата функции активации
        if (getIndexOfMaxElem(test_y[j][0][0], test_y[j][0][0] + 10) != 4) {
            // Если текущая цифра не 4, то ожидаемый ответ 0
            a[0][0] = 0;
        } else {
            // Если текущая цифра 4, то ожидаемый ответ 1
            a[0][0] = 1;
        }
        // Вычисление метрик
        Metrics[0][j] = metric_function(binaryAccuracy, y, a);
        Metrics[1][j] = metric_function(rmsError, y, a);
    }
    // Вывод метрик в консоль
    cout << "accuracy: ";
    cout << mean(Metrics[0], 90);
    cout << " loss: " << mean(Metrics[1], 90) << endl;

    return 0;
}
