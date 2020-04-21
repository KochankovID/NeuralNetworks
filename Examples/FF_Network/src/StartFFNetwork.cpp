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
    D_Matrix Metrics(2, 10);

#ifdef Teach
    // Создание обучающей выборки
    Vector<D_Tensor> data_x(10);
    Vector<D_Tensor> data_y(10);
    for(int i = 0; i < 10; i++){
        data_x[i] = data_y[i] = D_Tensor(1, 15, 1);
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
        auto tmp = D_Tensor(1,10,1);
        tmp.Fill(0);
        tmp[0][0][int(data_y[i][0][0][0])] = 1;
        data_y[i] = tmp;
    }

    // Обучение сети
    long int epoch = 1200; // Количество эпох
    cout << "Learning model: "<< endl;
    for (long int i = 0; i < epoch; i++) {
        for (int j = 0; j < 10; j++) { // Проход по обучающей выборке
            output = layer1.passThrough(data_x[j]); // Проход через слой нейронов
            error = loss_function(rmsErrorD, output[0], data_y[j][0]); // Вычисление ошибки
            Metrics[0][j] = metric_function(accuracy, output[0], data_y[j][0]); // Вычисление метрик
            Metrics[1][j] = metric_function(rmsError, output[0], data_y[j][0]);
            layer1.BackPropagation(error, data_x[j]); // Обратное распространение ошибки
            layer1.GradDes(G, data_x[j]); // Градиентный спуск
        }
        // Вывод метрик в консоль
        cout << "accuracy: ";
        cout <<  mean(Metrics[0], 10);
        cout << " loss: " << mean(Metrics[1], 10) << endl;
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
    Vector<D_Tensor> test_x(90);
    Vector<D_Tensor> test_y(90);
    for(int i = 0; i < 90; i++){
        test_x[i] = D_Tensor(1, 15, 1);
        test_y[i] = D_Tensor(1, 1, 1);
    }

    // Считывание тестовой выборки из файла
    io::CSVReader<17> in_test("./resources/test_nums.csv");
    int t;
    for(int i = 0; i < 90; i++){
        in_test.read_row(t,  test_x[i][0][0][0],
                         test_x[i][0][0][1], test_x[i][0][0][2],
                         test_x[i][0][0][3], test_x[i][0][0][4],
                         test_x[i][0][0][5], test_x[i][0][0][6],
                         test_x[i][0][0][7], test_x[i][0][0][8],
                         test_x[i][0][0][9], test_x[i][0][0][10],
                         test_x[i][0][0][11], test_x[i][0][0][12],
                         test_x[i][0][0][13], test_x[i][0][0][14], test_y[i][0][0][0]);
        auto tmp = D_Tensor(1,10,1);
        tmp.Fill(0);
        tmp[0][0][int(test_y[i][0][0][0])] = 1;
        test_y[i] = tmp;
    }

    Metrics = D_Matrix(2, 90);

    // Вывод в консоль реультатов тестирования сети на тестовой выборке
    cout << endl << "Validation model: " << endl;
    for (int j = 0; j < 90; j++) { // Проход по тестовой выборке
        output = layer1.passThrough(test_x[j][0]); // Проход через слой нейронов
        error = loss_function(rmsErrorD, output[0], test_y[j][0]); // Вычисление ошибки
        Metrics[0][j] = metric_function(accuracy, output[0], test_y[j][0]); // Вычисление метрик
        Metrics[1][j] = metric_function(rmsError, output[0], test_y[j][0]);
    }
    // Вывод метрик в консоль
    cout << "accuracy: ";
    cout <<  mean(Metrics[0], 90);
    cout << " loss: " << mean(Metrics[1], 90) << endl;

    return 0;
}