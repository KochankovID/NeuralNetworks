//: Нейросеть распознающая все цифры

#include "DenceLayer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include "csv.h"

// Макрос режима работы программы (с обучением или без)
#define Teach

// Поиск номера максимального элемента в массиве

using namespace std;
using namespace NN;

int main()
{

    // Создание функции активации
    D_Sigm  F(1);
    D_SigmD  f(1);

    // Создание производной функции ошибки
    D_RMS_errorD rmsErrorD;

    // Создание метрик
    D_RMS_error rmsError;
    D_Accuracy accuracy;

    // Создание градиентного спуска
    D_SGD G(0.09);

    // Создание инициализатора
    SimpleInitializator<double> I(1);

    // Создание весов нейросети
    D_DenceLayer layer1(10, 15, make_shared<D_Sigm>(F), make_shared<D_Sigm>(F), I);

    D_Tensor output;
    D_Tensor error;
    Ndarray<double> Metrics({2, 10});

#ifdef Teach
    // Создание обучающей выборки
    Ndarray<double> data_x({10, 15});
    Ndarray<double> data_y({10, 10});

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
        auto tmp = Ndarray<double>({10});
        tmp.fill(0);
        tmp[int(data_y(i,0))] = 1;
        std::copy(tmp.begin(), tmp.end(), data_y.iter(1, i, 0));
    }

    // Обучение сети
    long int epoch = 1200; // Количество эпох
    cout << "Learning model: "<< endl;
    for (long int i = 0; i < epoch; i++) {
        for (int j = 0; j < 10; j++) { // Проход по обучающей выборке
            output = layer1.passThrough(data_x.subArray(1, j)); // Проход через слой нейронов
            error = loss_function<double >(rmsErrorD, output[0], data_y.subArray(1,j)); // Вычисление ошибки
            Metrics(0,j) = metric_function<double >(accuracy, output[0], data_y.subArray(1,j)); // Вычисление метрик
            Metrics(1,j) = metric_function<double >(rmsError, output[0], data_y.subArray(1,j)); // Вычисление метрик
            layer1.BackPropagation(error, data_x.subArray(1,j)); // Обратное распространение ошибки
            layer1.GradDes(G, data_x.subArray(1,j)); // Градиентный спуск
        }
        // Вывод метрик в консоль
        cout << "accuracy: ";
        cout <<  Metrics.subArray(1,0).mean();
        cout << " loss: " << Metrics.subArray(1, 1).mean() << endl;
    }

    // Сохраняем веса
    ofstream file_out("./resources/Weights.txt");
    if(!file_out.is_open()){
        cout << "File openning error!" << endl;
        return 1;
    }
    layer1.saveToFile(file_out);

#else
    // Считывание весов
    ifstream file_out("./resources/Weights.txt");
    if(!file_out.is_open()){
        cout << "File openning error!" << endl;
        return 1;
    }
    layer1.getFromFile(file_out);

#endif // Teach
    // Создание тестовой выборки
    Ndarray<double > test_x({90,15});
    Ndarray<double> test_y({90,10});

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
        auto tmp = Ndarray<double>({10});
        tmp.fill(0);
        tmp[int(test_y(i,0))] = 1;
        std::copy(tmp.begin(), tmp.end(), test_y.iter(1, i, 0));
    }

    Metrics = Ndarray<double>({2, 90});

    // Вывод в консоль реультатов тестирования сети на тестовой выборке
    cout << endl << "Validation model: " << endl;
    for (int j = 0; j < 90; j++) { // Проход по тестовой выборке
        output = layer1.passThrough(test_x.subArray(1,j)); // Проход через слой нейронов
        error = loss_function<double>(rmsErrorD, output[0], test_y.subArray(1,j)); // Вычисление ошибки
        Metrics(0,j) = metric_function<double>(accuracy, output[0], test_y.subArray(1,j)); // Вычисление метрик
        Metrics(1,j) = metric_function<double>(rmsError, output[0], test_y.subArray(1,j));
    }
    // Вывод метрик в консоль
    cout << "accuracy: ";
    cout <<  Metrics.subArray(1,0).mean();
    cout << " loss: " << Metrics.subArray(1,1).mean() << endl;

    return 0;
}