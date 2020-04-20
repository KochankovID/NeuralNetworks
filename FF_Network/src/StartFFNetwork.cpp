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
using namespace ANN;

int main()
{

    // Создание функтора
    Sigm<double >  F(1);
    SigmD<double >  f(1);

    RMS_errorD<double> rmsErrorD;

    RMS_error<double> rmsError;
    Accuracy<double> accuracy;

    // Создание функтора
    SGD<double> G(0.09);

    SimpleInitializator<double> I(1);
    // Создание весов нейросети
    D_DenceLayer layer1(10, 15,F , f, I);


    Tensor<double> output;
    Tensor<double> error;
    Matrix<double> Metrics(2, 10);

#ifdef Teach

    // Последовательность цифр, тасуемая для получения равномерной рандомизации
    // Создание обучающей выборки
    Matrix<Tensor<double>> data_x(1, 10);
    Matrix<Tensor<double>> data_y(1, 10);
    for(int i = 0; i < 10; i++){
        data_x[0][i] = data_y[0][i] = Tensor<double>(1, 15, 1);
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

    // Обучение сети
    long int epoch = 1200; // Количество обучений нейросети
    cout << "Learning model: "<< endl;
    for (long int i = 0; i < epoch; i++) {
        for (int j = 0; j < 10; j++) { // Проход по обучающей выборке
            output = layer1.passThrough(data_x[0][j]);
            error = loss_function(rmsErrorD, output[0], data_y[0][j][0]);
            Metrics[0][j] = metric_function(accuracy, output[0], data_y[0][j][0]);
            Metrics[1][j] = metric_function(rmsError, output[0], data_y[0][j][0]);
            layer1.BackPropagation(error, data_x[0][j]);
            layer1.GradDes(G, data_x[0][j]);
        }
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

    // Вывод на экран реультатов тестирования сети на тестовой выборке
    Metrics = Matrix<double>(2, 90);
    cout << endl << "Validation model: " << endl;
    for (int j = 0; j < 90; j++) { // Проход по обучающей выборке
        output = layer1.passThrough(test_x[0][j]);
        error = loss_function(rmsErrorD, output[0], test_y[0][j][0]);
        Metrics[0][j] = metric_function(accuracy, output[0], test_y[0][j][0]);
        Metrics[1][j] = metric_function(rmsError, output[0], test_y[0][j][0]);
    }
    cout << "accuracy: ";
    cout <<  mean(Metrics[0], 90);
    cout << " loss: " << mean(Metrics[1], 90) << endl;

    return 0;
}