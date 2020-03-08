//: Нейросеть распознающая все цифры

#include "DenceNeyron.h"
#include "Data.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <random>

// Макрос режима работы программы (с обучением или без)
#define Teach

// Поиск номера максимального элемента в массиве
int max(const double* arr, const int& length) {
	int m = 0;
	for (int i = 1; i < length; i++) {
		if (arr[i] > arr[m]) {
			m = i;
		}
	}
	return m;
};

using namespace std;
using namespace ANN;

int main()
{

	// Создание функтора
	Sigm<double >  F(6);
    RMS_error<double> MM;

    // Создание функтора
    Accuracy<double> M;

	// Создание весов нейросети
	Matrix<Neyron<double>> W(1,10);
	for (int i = 0; i < 10; i++) {
		W[0][i] = Neyron<double>(5, 3);
		W[0][i].Fill(1);
	}

	double summ; // Переменная суммы
	int y; // Переменная выхода сети

    vector<double> output(10);
    vector<double> correct(10);

#ifdef Teach

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	int nums[10] = { 0,1,2,3,4,5,6,7,8,9 };
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	// Создание обучающей выборки
	Matrix<Matrix<double>> Nums(1, 10);

	double error;

#define NUMBER nums[j]

	// Считываем матрицы обучающей выборки
	getDataFromTextFile(Nums, "./resources/TeachChoose.txt");

	// Обучение сети
	long int epoch = 20; // Количество обучений нейросети

	for (long int i = 0; i < epoch; i++) {
//		shuffle(nums, nums+10, default_random_engine(seed)); // Тасование последовательности
		for (int j = 0; j < 10; j++) { // Проход по обучающей выборке
			for (int l = 0; l < 10; l++) { // Проход по всем нейронам сети
				summ = W[0][l].Summator(Nums[0][NUMBER]); // Получение взвешенной суммы
				y = D_Neyron::FunkActiv(summ, F); // Получение ответа нейрона
				if (NUMBER != l) {
					// Если номер текущего нейрона не совпадает с текущей цифрой, то ожидаемый ответ 0
					SimpleLearning<double>(0.0, y, W[0][l], Nums[0][NUMBER]);
					output[j] = y;
					correct[j] = 0;
				}
				else {
					// Если номер текущего нейрона совпадает с текущей цифрой, то ожидаемый ответ 1
                    SimpleLearning<double>(1.0, y, W[0][l], Nums[0][NUMBER]);
                    output[j] = y;
                    correct[j] = 1;
				}
			}
			cout << "||";
		}
        error = loss_function(MM, output, correct);
        cout << "] accuracy: ";
        cout << metric_function(M, output, correct);
        cout << " loss: " << error << endl;
	}

	// Сохраняем веса
	saveWeightsTextFile(W, "./resources/Weights.txt");

#else
	// Считывание весов
	getWeightsTextFile(W, "./resources/Weights.txt");

#endif // Teach

	// Создание тестовой выборки
	Matrix<Matrix<double>> Tests(1, 20);

	// Считывание тестовой выборки из файла
	getDataFromTextFile(Tests, "./resources/Tests.txt");

	// Вывод на экран реультатов тестирования сети на обучающей выборке
	cout << "Test network:" << endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			summ = W[0][j].Summator(Tests[0][i]); // Получение взвешенной суммы
			output[j] = D_Neyron::FunkActiv(summ, F);
		}

		y = max_element(output.begin(), output.end()) - output.begin();
		// Вывод результатов на экран
		cout << "Test " << i << " : " << "recognized " << y << endl;
	}

	// Вывод на экран реультатов тестирования сети на тестовой выборке
	cout << "Test resilience:" << endl;
	for (int i = 10; i < 20; i++) {
		for (int j = 0; j < 10; j++) {
            summ = W[0][j].Summator(Tests[0][i]); // Получение взвешенной суммы
            output[j] = D_Neyron::FunkActiv(summ, F);
		}
        y = max_element(output.begin(), output.end()) - output.begin();
		// Вывод результатов на экран
		cout << "Test " << i << " : " << "recognized " << y << endl;
	}

	// Вывод весов сети на экран
	cout << endl << "Weights of network: " << endl;
	for (int i = 0; i < 10; i++) {
		cout << "Weight " << i << "-th neyron's:" << endl;
		W[0][i].Out();
		cout << endl;
	}

	return 0;

}