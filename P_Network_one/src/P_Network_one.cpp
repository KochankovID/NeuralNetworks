//: Нейросеть распознающая 4

#include "Neyron.h"
#include "Initializers.h"
#include <vector>
#include <iostream>
#include <fstream>
#include "Data.h"
#include <algorithm>
#include <random>
#include "csv.h"

// Макрос режима работы программы (с обучением или без)
#define Teach


using namespace std;
using namespace ANN;
int main()
{
	// Создание функции активации
    BinaryClassificator<int> F;
    BinaryClassificatorD<int> FD;

    // Создание функции ошибки
    RMS_error<int> MM;

	// Создание метрики
	Accuracy<int> M;

	// Создание инициализатора весов
	SimpleInitializator<int> I( 1);

	// Создание cлоя нейрона из одного нейрона
	I_Neyron neyron(1, 10);
    
#ifdef Teach
    // Создание обучающей выборки
    Matrix<Tensor<int>> data_x(1, 10);
    Matrix<Tensor<int>> data_y(1, 10);
    for(int i = 0; i < 10; i++){
        data_x[0][i] = data_y[0][i] = Tensor<int>(1, 15, 1);
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

        auto tmp = Tensor<int >(1,10,1);
        tmp.Fill(0);
        tmp[0][0][int(data_y[0][i][0][0][0])] = 1;
        data_y[0][i] = tmp;
    }

	// Обучение сети
	long int epoch = 8; // Количество обучений нейросети
	Matrix<int> y(1,1); // Переменная выхода сети
	Matrix<int> a(1,1);
	Matrix<int> output(10,1);
    Matrix<int> correct(10,1);

    int summ;
    int out;
	for (long int i = 0; i < epoch; i++) {
		for (int j = 0; j < 10; j++) {
		    summ = neyron.Summator(data_x[0][j][0]);
		    out = neyron.FunkActiv(summ, F);
			if (j != 4) {
				// Если текущая цифра не 4, то ожидаемый ответ 0
				a[0][0] = 0;
				SimpleLearning(a, out, neyron, data_x[0][j])
				output[j][0] = y[0][0];
				correct[j][0] = 0;
			}
			else {
				// Если текущая цифра 4, то ожидаемый ответ 1
                a[0][0] = 1;
                layer.SimpleLearning(a, y, Nums[0][j], 1);
                output[j][0] = y[0][0];
                correct[j][0] = 1;
            }
            cout << "||";
		}
        cout << "] accuracy: ";
        cout << metric_function(M, output, correct);
        cout << " loss: " << metric_function(MM, output, correct) << endl;

    }

    // Проверка работы сети на обучающей выборке
    // Вывод результатов на экран
    cout << "Test network:" << endl;
    for (int i = 0; i < 10; i++) {
        // Вывод результатов на экран
        layer.passThrough(Nums[0][i])[0][0] == 1 ? cout << "Test " << i << " : " << "recognized 4" << endl : cout << "Test " << 0 << " : " << "doesn't recognized 4" << endl;
    }

	// Сохраняем веса
	layer.saveWeightsToFile("./resources/Weights.txt");

#else
	// Считывание весов
	layer.getWeightsFromFile("./resources/Weights.txt");

#endif // Teach
	// Создание тестовой выборки
	Matrix<Matrix<int>> Tests(1,9);

	// Считывание тестовой выборки из файла
	getDataFromTextFile(Tests, "./resources/Tests.txt");

	// Вывод на экран реультатов тестирования сети на тестовой выборке
	cout << "Test resilience:" << endl;
	for (int i = 0; i < Tests.getM(); i++) {
		// Вывод результатов на экран
        layer.passThrough(Tests[0][i])[0][0] == 1 ? cout << "Test " << i << " : " << "recognized 4" << endl : cout << "Test " << 0 << " : " << "doesn't recognized 4" << endl;
	}

	// Вывод весов сети
	cout << endl << "Weights of network: " << endl;
	cout << layer;
	return 0;
}
