//: Нейросеть распознающая 4

#include "DenceNeyron.h"
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
	// Создание градиентного спуска
    SimpleGrad<int> G(1);

    RMS_error<double> MM;

	// Создание функтора
	BinaryClassificator<int> F;
	Accuracy<double> M;

	// Создание весов нейрона
	I_Neyron neyron(5,3);
#ifdef Teach
    int nums[] = {0,1,2,3,4,5,6,7,8,9};
    neyron.Fill(1);

	// Создание обучающей выборки
	Matrix<Matrix<int>> Nums(1,10);

	// Создание обучающей выборки
	// Считываем матрицы обучающей выборки
	getDataFromTextFile(Nums, "./resources/TeachChoose.txt");

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	// Обучение сети
	long int epoch = 7; // Количество обучений нейросети
	int summ; // Переменная суммы
	int y; // Переменная выхода сети
	double error;
	Matrix<double> output(1,10);
    Matrix<double> correct(1,10);
    Matrix<double > losses_on_batch(1,10);
    Matrix<double > accurency_on_batch(1, 10);
    double accurency;

#define NUMBER nums[j]

	for (long int i = 0; i < epoch; i++) {
        shuffle(nums, nums+10, default_random_engine(seed));
		for (int j = 0; j < 10; j++) {
			summ = neyron.Summator(Nums[0][NUMBER]); // Получение взвешенной суммы
			y = I_Neyron::FunkActiv(summ, F); // Получение ответа нейрона
			if (NUMBER != 4) {
				// Если текущая цифра не 4, то ожидаемый ответ 0
				SimpleLearning(0,y,neyron, Nums[0][NUMBER], 1);
				output[0][j] = y;
				correct[0][j] = 0;
			}
			else {
				// Если текущая цифра 4, то ожидаемый ответ 1
                SimpleLearning(1,y,neyron, Nums[0][NUMBER],1);
                output[0][j] = y;
                correct[0][j] = 1;
            }
            losses_on_batch[0][j] = loss_function(MM, output, correct);
            accurency_on_batch[0][j] = metric_function(M, output, correct);
            cout << "||";
		}
        error = 0;
        accurency = 0;
        for(size_t ii = 0; ii < 10; ii++){
            error += losses_on_batch[0][ii];
            accurency += accurency_on_batch[0][ii];
        }
        error /= 10;
        accurency /= 10;
        cout << "] accuracy: ";
        cout << accurency;
        cout << " loss: " << error << endl;

    }

    // Проверка работы сети на обучающей выборке
    // Вывод результатов на экран
    cout << "Test network:" << endl;
    for (int i = 0; i < 10; i++) {
        // Вывод результатов на экран
        I_Neyron::FunkActiv(neyron.Summator(Nums[0][i]), F) == 1 ? cout << "Test " << i << " : " << "recognized 4" << endl : cout << "Test " << 0 << " : " << "doesn't recognized 4" << endl;
    }

	// Сохраняем веса
	saveWeightsTextFile(neyron,"./resources/Weights.txt" );

#else
	// Считывание весов
	getWeightsTextFile(neyron, "Weights.txt");

#endif // Teach
	// Создание тестовой выборки
	Matrix<Matrix<int>> Tests(1,14);

	// Считывание тестовой выборки из файла
	getDataFromTextFile(Tests, "./resources/Tests.txt");

	// Вывод на экран реультатов тестирования сети на тестовой выборке
	cout << "Test resilience:" << endl;
	for (int i = 0; i < 14; i++) {
		// Вывод результатов на экран
        I_Neyron::FunkActiv(neyron.Summator(Tests[0][i]), F) == 1 ? cout << "Test " << i << " : " << "recognized 4" << endl : cout << "Test " << 0 << " : " << "doesn't recognized 4" << endl;
	}

	// Вывод весов сети
	cout << endl << "Weights of network: " << endl;
	neyron.Out();
	return 0;
}
