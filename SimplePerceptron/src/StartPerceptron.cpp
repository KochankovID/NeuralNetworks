//: Нейросеть распознающая 4

#include "DenceLayers.h"
#include <vector>
#include <iostream>
#include <fstream>
#include "Data.h"
#include <algorithm>
#include <random>

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
	allOne<int> I( -1);

	// Создание cлоя нейрона из одного нейрона
	I_DenceLayer layer(1, 15, F, FD, I);

#ifdef Teach
    int nums[] = {0,1,2,3,4,5,6,7,8,9};

	// Создание обучающей выборки
	Matrix<Matrix<int>> Nums(1,10);

	// Создание обучающей выборки
	// Считываем матрицы обучающей выборки
	getDataFromTextFile(Nums, "./resources/TeachChoose.txt");

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	// Обучение сети
	long int epoch = 8; // Количество обучений нейросети
	Matrix<int> y(1,1); // Переменная выхода сети
	Matrix<int> a(1,1);
	Matrix<int> output(10,1);
    Matrix<int> correct(10,1);


	for (long int i = 0; i < epoch; i++) {
		for (int j = 0; j < 10; j++) {
			y = layer.passThrough(Nums[0][j]);
			if (j != 4) {
				// Если текущая цифра не 4, то ожидаемый ответ 0
				a[0][0] = 0;
				layer.SimpleLearning(a, y, Nums[0][j], 1);
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
	layer.Out();
	return 0;
}
