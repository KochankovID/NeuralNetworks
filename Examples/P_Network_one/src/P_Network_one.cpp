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
    Ndarray<double> Metrics({2, 10});
    I_Matrix y(1, 1);
    I_Matrix a(1, 1);
    I_Vector output(10);
#ifdef Teach
    // Создание обучающей выборки
    Ndarray<int> data_x({10,15});
    Ndarray<int> data_y({10, 10});

    // Считываем матрицы обучающей выборки
    io::CSVReader<16> in("./resources/training_nums.csv");
    for (int i = 0; i < 10; i++) {
        in.read_row(data_y(i,0), data_x(i,0),
                    data_x(i,1), data_x(i,2),
                    data_x(i,3), data_x(i,4),
                    data_x(i,5), data_x(i,6),
                    data_x(i,7), data_x(i,8),
                    data_x(i,9), data_x(i,10),
                    data_x(i,11), data_x(i,12),
                    data_x(i,13), data_x(i,14));
        auto tmp = Ndarray<int>({10});
        tmp.fill(0);
        tmp[int(data_y(i,0))] = 1;
        std::copy(tmp.begin(), tmp.end(), data_y.iter(1, i, 0));
    }

    // Обучение сети
    long int epoch = 6; // Количество обучений нейросети
    for (long int i = 0; i < epoch; i++) {
        for (int j = 0; j < 10; j++) { // Проход по обучающей выборке
            summ = neyron.Summator(data_x.subArray(1, j)); // Вычисление взвешенной суммы
            y[0][0] = neyron.FunkActiv(summ, F); // Получение результата функции активации
            if (j != 4) {
                // Если текущая цифра не 4, то ожидаемый ответ 0
                a[0][0] = 0;
            } else {
                // Если текущая цифра 4, то ожидаемый ответ 1
                a[0][0] = 1;
            }
            SimpleLearning<int>(a[0][0], y[0][0], neyron, data_x.subArray(1,j), 1); // Калибровка весов нейрона
            cout << "||";
            // Вычисление метрик
            Metrics(0,j) = metric_function(binaryAccuracy, y, a);
            Metrics(1,j) = metric_function(rmsError, y, a);
        }
        // Вывод метрик в консоль
        cout << " accuracy: ";
        cout << Metrics.subArray(1,0).mean();
        cout << " loss: " << Metrics.subArray(1,1).mean() << endl;
    }

    // Сохраняем веса
    saveWeightsTextFile(neyron, "./resources/Weights.txt");
#else
    // Считывание весов
    getWeightsTextFile(neyron, "./resources/Weights.txt");

#endif // Teach
    // Создание тестовой выборки
    Ndarray<int> test_x({90,15});
    Ndarray<int> test_y({90,10});

    // Считывание тестовой выборки из файла
    io::CSVReader<17> in_test("./resources/test_nums.csv");
    int t;
    for (int i = 0; i < 90; i++) {
        in_test.read_row(t, test_x(i,0),
                         test_x(i,1), test_x(i,2),
                         test_x(i,3), test_x(i,4),
                         test_x(i,5), test_x(i,6),
                         test_x(i,7), test_x(i,8),
                         test_x(i,9), test_x(i,10),
                         test_x(i,11), test_x(i,12),
                         test_x(i,13), test_x(i,14),
                         test_y(i,0));
        auto tmp = Ndarray<int>({10});
        tmp.fill(0);
        tmp[int(test_y(i,0))] = 1;
        std::copy(tmp.begin(), tmp.end(), test_y.iter(1, i, 0));
    }

    Metrics = Ndarray<double>({2, 90});

    // Вывод на экран реультатов тестирования сети на тестовой выборке
    cout << endl << "Validation model: " << endl;
    for (int j = 0; j < 90; j++) { // Проход по тестовой выборке
        summ = neyron.Summator(test_x.subArray(1,j)); // Вычисление взвешенной суммы
        y[0][0] = neyron.FunkActiv(summ, F); // Получение результата функции активации
        if (test_y.subArray(1,j).argmax() != 4) {
            // Если текущая цифра не 4, то ожидаемый ответ 0
            a[0][0] = 0;
        } else {
            // Если текущая цифра 4, то ожидаемый ответ 1
            a[0][0] = 1;
        }
        // Вычисление метрик
        Metrics(0,j) = metric_function(binaryAccuracy, y, a);
        Metrics(1,j) = metric_function(rmsError, y, a);
    }
    // Вывод метрик в консоль
    cout << "accuracy: ";
    cout << Metrics.subArray(1, 0).mean();
    cout << " loss: " << Metrics.subArray(1, 1).mean() << endl;

    return 0;
}
