//: Нейросеть распознающая все цифры

#include "Neyrons.h"
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
	Sigm<double > F(0.9);

    RMS_error<double> MM;
    Accuracy<double> M;
	RMS_errorD<double> R;

	// Создание производной функтора
	SigmD<double > f(0.9);

	// Установка зерна для выдачи рандомных значений
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Количество нейронов первого слоя нейросети
	const int w1_count = 50;

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

	SimpleGrad<double> G(0.8);
	double error;
	double accuracy;
	Matrix<double> error_vect(1, 10);
	// Создание обучающей выборки
	Matrix<Matrix<double>> Nums(1, 10);
	// Считываем матрицы обучающей выборки
	getDataFromTextFile(Nums, "./resources/TeachChoose.txt");

	// Обучение сети
	long int k = 400; // Количество обучений нейросети

	for (long int i = 1; i < k; i++) {
//        shuffle(nums, nums+10, default_random_engine(seed)); // Тасование последовательности
		error = 0;
		accuracy = 0;
		for (int j = 0; j < 10; j++) { // Цикл прохода по обучающей выборке
				for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
					summ = W[0][l].Summator(Nums[0][NUMBER]); // Получение взвешенной суммы
					m[0][l] = D_Neyron::FunkActiv(summ, F);
				}
				for (int l = 0; l < 10; l++) { // Цикл прохода по сети
					summ = W1[0][l].Summator(m); // Получение взвешенной суммы
					output[j][l] = D_Neyron::FunkActiv(summ, F); // Запись выхода l-того нейрона в массив выходов сети
                    if (l == NUMBER) {
                        correct[j][l] = 1;
                    }else{
                        correct[j][l] = 0;
                    }

				}

                error_vect = loss_function(R, output.getPodmatrix(j,0,1,10), correct.getPodmatrix(j,0,1,10));

				for (int l = 0; l < 10; l++) { // Расчет ошибки для выходного слоя
						W1[0][l].GetD() = error_vect[0][l]; // Расчет ошибки
				}

				for (int l = 0; l < 10; l++) { // Распространение ошибки на скрытые слои нейросети
					BackPropagation(W, W1[0][l]);
				}
				int dropout1 = rand()*10;
				int dropout2 = rand()*10;
				for (int l = 0; l < w1_count; l++) { // Примемение градиентного спуска по всем нейроннам первого слоя
				    if((dropout1 == l)||(dropout2 == l)) continue;
					GradDes(G, W[0][l], Nums[0][NUMBER], f);
				}
                dropout1 = rand()*10;
                dropout2 = rand()*10;
				for (int l = 0; l < 10; l++) { // Примемение градиентного спуска по всем нейроннам второго слоя
                    if((dropout1 == l)||(dropout2 == l)) continue;
					GradDes(G, W1[0][l], m, f);
				}
				for(int l = 0; l < 10; l++){
				    W[0][l].GetD() = 0;
				}
				// "Стягивание весов"
//				retract(W, 4);
//				retract(W1, 4);
			cout << "||";
		}
        cout << "] accuracy: ";
        cout << metric_function(M, output, correct);
        cout << " loss: " << metric_function(MM, output, correct) << endl;
	}

	// Сохраняем веса
	saveWeightsTextFile(W, "./resources/Weights0.txt");
    saveWeightsTextFile(W1, "./resources/Weights1.txt");

#else
	// Считывание весов
	getWeightsTextFile(W, "./resources/Weights0.txt");
	getWeightsTextFile(W1, "./resources/Weights1.txt");

#endif // Teach

	// Создание тестовой выборки
	Matrix<Matrix<double>> Tests(1, 25);

	// Считывание тестовой выборки из файла
	getDataFromTextFile(Tests, "./resources/Tests.txt");

	Matrix<double> out(1, 10);
	// Вывод на экран реультатов тестирования сети
	cout << "Test network:" << endl;
	for (int j = 0; j < 10; j++) { // Цикл прохода по тестовой выборке
		for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
            summ = W[0][l].Summator(Tests[0][j]); // Получение взвешенной суммы
            m[0][l] = D_Neyron::FunkActiv(summ, F);
		}
		for (int l = 0; l < 10; l++) { // Цикл прохода по сети
            summ = W1[0][l].Summator(m); // Получение взвешенной суммы
            out[0][l] = D_Neyron::FunkActiv(summ, F); // Запись выхода l-того нейрона в массив выходов сети
		}
        int max = max_element(out[0], out[0]+10) - out[0];
		// Вывод результатов на экран
		cout << "Test " << j << " : " << "recognized " << max << ' ' << out[0][max] << endl;
	}

	cout << "Test resilience:" << endl;
	for (int j = 10; j < 24; j++) { // Цикл прохода по тестовой выборке
		for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
            summ = W[0][l].Summator(Tests[0][j]); // Получение взвешенной суммы
            m[0][l] = D_Neyron::FunkActiv(summ, F);
		}
		for (int l = 0; l < 10; l++) { // Цикл прохода по сети
            summ = W1[0][l].Summator(m); // Получение взвешенной суммы
            out[0][l] = D_Neyron::FunkActiv(summ, F); // Запись выхода l-того нейрона в массив выходов сети
		}
        int max = max_element(out[0], out[0]+10) - out[0];
        // Вывод результатов на экран
        cout << "Test " << j << " : " << "recognized " << max << ' ' << out[0][max] << endl;

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