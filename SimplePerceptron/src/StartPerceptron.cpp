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

	// Создание функтора
	Sigm<int> F(1);
    SigmD<int> f(1);

	// Создание весов нейрона
	I_Neyron neyron(5,3);
#ifdef Teach
    int nums[] = {0,1,2,3,4,5,6,7,8,9};

	// Создание обучающей выборки
	vector<Matrix<int>> Nums(10);

	// Создание обучающей выборки
	// Считываем матрицы обучающей выборки
	getDataFromTextFile(Nums, "./resources/TeachChoose.txt");

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	// Обучение сети
	long int epoch = 167; // Количество обучений нейросети
	int summ; // Переменная суммы
	int y; // Переменная выхода сети
#define NUMBER nums[j]

	for (long int i = 0; i < epoch; i++) {

        shuffle(nums, nums+9, default_random_engine(seed));
		for (int j = 0; j < 10; j++) {
			summ = neyron.Summator(Nums[NUMBER]); // Получение взвешенной суммы
			y = I_Neyron::FunkActiv(summ, F); // Получение ответа нейрона
			if (NUMBER != 4) {
				// Если текущая цифра не 4, то ожидаемый ответ 0
				neyron.GetD() = 0 - y;
			}
			else {
				// Если текущая цифра 4, то ожидаемый ответ 1
                neyron.GetD() = 1 - y;
			}
			GradDes(G, neyron, Nums[NUMBER], f);
		}
	}

	// Сохраняем веса
	saveWeightsTextFile(neyron,"./resources/Weights.txt" );

#else
	// Считывание весов
	getWeightsTextFile(neyron, "Weights.txt");

#endif // Teach
	// Создание тестовой выборки
	vector<Matrix<int>> Tests(14);


	// Считывание тестовой выборки из файла
	getDataFromTextFile(Tests, "./resources/Tests.txt");

	// Проверка работы сети на обучающей выборке
	cout << "Test network:" << endl;
	// Вывод результатов на экран
	I_Neyron::FunkActiv(neyron.Summator(Tests[0]), F) == 1 ? cout << "Test " << 0 << " : " << "recognized 4" << endl : cout << "Test " << 0 << " : " << "doesn't recognized 4" << endl;

	// Вывод на экран реультатов тестирования сети на тестовой выборке
	cout << "Test resilience:" << endl;
	for (int i = 0; i < 4; i++) {
		// Вывод результатов на экран
        I_Neyron::FunkActiv(neyron.Summator(Tests[0]), F) == 1 ? cout << "Test " << 0 << " : " << "recognized 4" << endl : cout << "Test " << 0 << " : " << "doesn't recognized 4" << endl;
	}

	// Вывод весов сети
	cout << endl << "Weights of network: " << endl;
	neyron.Out();
	return 0;
}
