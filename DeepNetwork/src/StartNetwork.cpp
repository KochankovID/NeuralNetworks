//: Нейросеть распознающая все цифры

#include "DenceNeyron.h"
#include "Data.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <math.h>
#include <opencv2/ml.hpp>

// Макрос режима работы программы (с обучением или без)
#define Teach

// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;
using namespace ANN;

int main()
{
	// Создание функтора
	Sigm<double > F(3.4);
	Relu<double >F1(1);

    RMS_error<double> MM;
    Accuracy<double> M;

	// Создание производной функтора
	SigmD<double > f(3.4);
	ReluD<double > f1(1);

	// Установка зерна для выдачи рандомных значений
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Количество нейронов первого слоя нейросети
	const int w1_count = 100;

    Matrix<double> output(10, 10);
    Matrix<double> correct(10, 10);

	// Создание весов нейросети
	Matrix<Neyron<double>> W(1, w1_count);
	for (int i = 0; i < w1_count; i++) {
		W[0][i] = Neyron<double>(5, 3);
		for (int j = 0; j < W[0][i].getN(); j++) {
			for (int p = 0; p < W[0][i].getM(); p++) {
				W[0][i][j][p] = (p % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
			}
		}
		W[0][i].GetWBias() = (i % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
	}

	// Создания весов для второго слоя сети
	Matrix<Neyron<double>> W1(1, 10);
	for (int i = 0; i < 10; i++) {
		W1[0][i] = Neyron<double>(1, w1_count);
		for (int j = 0; j < W1[0][i].getN(); j++) {
			for (int p = 0; p < W1[0][i].getM(); p++) {
				W1[0][i][j][p] = (p % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
			}
		}
		W1[0][i].GetWBias() = (i % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
	}

	// Массив выходов первого слоя сети
	Matrix<double> m(1, w1_count);

	double summ; // Переменная суммы

#ifdef Teach

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	int nums[10] = { 0,1,2,3,4,5,6,7,8,9 };

	// Создание обучающей выборки
	Matrix<Matrix<double>> Nums(1, 10);
	// Считываем матрицы обучающей выборки
	getDataFromTextFile(Nums, "./resources/TeachChoose.txt");

	// Обучение сети
	long int k = 70; // Количество обучений нейросети

	for (long int i = 1; i < k; i++) {
        shuffle(nums, nums+10, default_random_engine(seed)); // Тасование последовательности
		cout << i << endl;
		for (int j = 0; j < 10; j++) { // Цикл прохода по обучающей выборке
			for (int u = 0; u < 3; u++) {
				for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
					summ = W[0][l].Summator(Nums[0][NUMBER]); // Получение взвешенной суммы
					m[0][l] = D_Neyron::FunkActiv(summ, F);
				}
				for (int l = 0; l < 10; l++) { // Цикл прохода по сети
					summ = W1[0][l].Summator(m); // Получение взвешенной суммы
					output[j][l] = D_Neyron::FunkActiv(summ, F); // Запись выхода l-того нейрона в массив выходов сети
				}

				for (int l = 0; l < 10; l++) { // Расчет ошибки для выходного слоя
					if (l == NUMBER) { // Если номер нейрона совпадает с поданной на вход цифрой, то ожидаеммый ответ 1
						W1[0][l].GetD() = Teacher.PartDOutLay(1, y[l]); // Расчет ошибки
					}
					else {// Если номер нейрона совпадает с поданной на вход цифрой, то ожидаеммый ответ 1
						W1[0][l].GetD() = Teacher.PartDOutLay(0, y[l]); // Расчет ошибки
					}
				}

				for (int l = 0; l < 10; l++) { // Распространение ошибки на скрытые слои нейросети
					Teacher.BackPropagation(W, W1[0][l]);
				}
				for (int l = 0; l < w1_count; l++) { // Примемение градиентного спуска по всем нейроннам первого слоя
					Teacher.GradDes(W[0][l], Nums[NUMBER], f, m[0][l]);
				}
				for (int l = 0; l < 10; l++) { // Примемение градиентного спуска по всем нейроннам второго слоя
					summ = Neyron.Summator(m, W1[0][l]);
					Teacher.GradDes(W1[0][l], m, f, summ);
				}
				for (int l = 0; l < w1_count; l++) { // Обнуление ошибки нейронов 1 слоя
					W[0][l].GetD() = 0;
				}
				// "Стягивание весов"
				Teacher.retract(W, 4);
				Teacher.retract(W1, 4);
			}
		}
	}

	// Сохраняем веса
	ofstream fWeights;
	fWeights.open("./resources/Weights.txt");
	fWeights << W;
	fWeights << W1;
	fWeights.close();

#else
	// Считывание весов
	ifstream fWeights;
	fWeights.open("Weights.txt");
	fWeights >> W;
	fWeights >> W1;
	fWeights.close();
#endif // Teach

	// Создание тестовой выборки
	vector<Matrix<double>> Tests(25);

	// Считывание тестовой выборки из файла
	ifstream Testsnums;
	Testsnums.open("./resources/Tests.txt");
	for (int i = 0; i < 24; i++) {
		Testsnums >> Tests[i];
	}
	double max;

	// Вывод на экран реультатов тестирования сети
	cout << "Test network:" << endl;
	for (int j = 0; j < 10; j++) { // Цикл прохода по тестовой выборке
		for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
			summ = Neyron.Summator(Tests[j], W[0][l]); // Получение взвешенной суммы
			m[0][l] = Neyron.FunkActiv(summ, F);
		}
		for (int l = 0; l < 10; l++) { // Цикл прохода по сети
			summ = Neyron.Summator(m, W1[0][l]); // Получение взвешенной суммы
			y[l] = Neyron.FunkActiv(summ, F); // Запись выхода l-того нейрона в массив выходов сети
		}
		int max = 0;
		for (int l = 1; l < 10; l++) { // Получение результатов сети
			if (y[l] > y[max]) {
				max = l;
			}
		}
		// Вывод результатов на экран
		cout << "Test " << j << " : " << "recognized " << max << ' ' << y[max] << endl;
	}

	cout << "Test resilience:" << endl;
	for (int j = 10; j < 24; j++) { // Цикл прохода по тестовой выборке
		for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
			summ = Neyron.Summator(Tests[j], W[0][l]); // Получение взвешенной суммы
			m[0][l] = Neyron.FunkActiv(summ, F);
		}
		for (int l = 0; l < 10; l++) { // Цикл прохода по сети
			summ = Neyron.Summator(m, W1[0][l]); // Получение взвешенной суммы
			y[l] = Neyron.FunkActiv(summ, F); // Запись выхода l-того нейрона в массив выходов сети
		}
		int max = 0;
		for (int l = 1; l < 10; l++) { // Получение результатов сети
			if (y[l] > y[max]) {
				max = l;
			}
		}
		// Вывод результатов на экран
		cout << "Test " << j << " : " << "recognized " << max << ' ' << y[max] << endl;
	}

	// Вывод весов сети на экран
	cout << endl << "Weights of network. First layer: " << endl;
	for (int i = 0; i < 10; i++) {
		cout << "Weight " << i << "-th neyron's:" << endl;
		W[0][i].Out();
		cout << endl;
	}
	cout << endl << "Weights of network. Second layer: " << endl;
	for (int i = 0; i < 10; i++) {
		cout << "Weight " << i << "-th neyron's:" << endl;
		W1[0][i].Out();
		cout << endl;
	}
	return 0;

}