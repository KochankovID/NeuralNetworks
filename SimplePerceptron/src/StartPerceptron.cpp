//: Нейросеть распознающая 4

#include "Perceptrons.h"
#include <vector>
#include <iostream>
#include <fstream>

// Макрос режима работы программы (с обучением или без)
#define Teach

// Функтор
class Sign : public I_Func
{
public:
	Sign() : I_Func(){};
	int operator()(const int& x) {
		if (x <= 0) {
			return -1;
		}
		else {
			return 1;
		}
	}
	~Sign() {};
};

using namespace std;
int main()
{
	// Создание перцептрона
	I_Perceptron Neyron;

	// Создание обучателя сети
	I_Leaning Teacher;

	// Создание функтора
	Sign F;

	// Создание весов нейрона
	Weights<int> Weight(5,3);
#ifdef Teach

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	int nums[10] = { 0,1,2,3,4,5,6,7,8,9 };

	// Создание обучающей выборки
	vector<Matrix<int>> Nums(10);

	// Создание обучающей выборки
	// Считываем матрицы обучающей выборки
	ifstream TeachChoose;

	TeachChoose.open("./resources/TeachChoose.txt");
	for (int i = 0; i < Nums.size(); i++) {
		TeachChoose >> Nums[i];
	}
	TeachChoose.close();

	// Обучение сети
	long int k = 167; // Количество обучений нейросети
	int summ; // Переменная суммы
	int y; // Переменная выхода сети

	for (long int i = 0; i < k; i++) {
		Teacher.shuffle(nums, 10);
		for (int j = 0; j < 10; j++) {
			summ = Neyron.Summator(Nums[nums[j]], Weight); // Получение взвешенной суммы
			y = Neyron.FunkActiv(summ, F); // Получение ответа нейрона
			if (nums[j] != 4) {
				// Если текущая цифра не 4, то ожидаемый ответ -1
				Teacher.WTSimplePerceptron(-1, y, Weight, Nums[nums[j]]);
			}
			else {
				// Если текущая цифра 4, то ожидаемый ответ 1
				Teacher.WTSimplePerceptron(1, y, Weight, Nums[nums[j]]);
			}
		}
	}

	// Сохраняем веса
	ofstream fWeights;
	fWeights.open("./resources/Weights.txt");
	fWeights << Weight;
	fWeights.close();

#else
	// Считывание весов
	ifstream fWeights;
	fWeights.open("Weights.txt");
	fWeights >> Weight;
	fWeights.close();

#endif // Teach
	// Создание тестовой выборки
	vector<Matrix<int>> Tests(14);


	// Считывание тестовой выборки из файла
	ifstream Testsnums;
	Testsnums.open("./resources/Tests.txt");
	for (int i = 0; i < 14; i++) {
		Testsnums >> Tests[i];
	}

	// Проверка работы сети на обучающей выборке
	cout << "Test network:" << endl;
	// Вывод результатов на экран
	Neyron.FunkActiv(Neyron.Summator(Tests[0], Weight), F) == 1 ? cout << "Test " << 0 << " : " << "recognized 4" << endl : cout << "Test " << 0 << " : " << "doesn't recognized 4" << endl;

	// Вывод на экран реультатов тестирования сети на тестовой выборке
	cout << "Test resilience:" << endl;
	for (int i = 0; i < 4; i++) {
		// Вывод результатов на экран
		Neyron.FunkActiv(Neyron.Summator(Tests[i], Weight), F) == 1 ? cout << "Test " << i << " : " << "recognized 4" << endl : cout << "Test " << i << " : " << "doesn't recognized 4" << endl;
	}

	// Вывод весов сети
	cout << endl << "Weights of network: " << endl;
	Weight.Out();
	return 0;
}
