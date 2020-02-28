//: Нейросеть распознающая все цифры

#include "Perceptrons.h"
#include <vector>
#include <iostream>
#include <fstream>

// Макрос режима работы программы (с обучением или без)
#define Teach

using namespace std;
int main()
{
	// Создание перцептрона
	D_Perceptron Neyron;

	// Создание обучателя сети
	D_Leaning Teacher;
	Teacher.getE() = 1;

	// Создание функтора
	Sigm<double > F(1);

	// Создание весов нейросети
	vector<Weights<double>> W(10);
	for (int i = 0; i < 10; i++) {
		W[i] = Weights<double>(5, 3);
	}

#ifdef Teach

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	int nums[10] = { 0,1,2,3,4,5,6,7,8,9 };

	// Создание обучающей выборки
	vector<Matrix<double>> Nums(10);

	// Считываем матрицы обучающей выборки
	ifstream TeachChoose;
	TeachChoose.open("./resources/TeachChoose.txt");
	for (int i = 0; i < Nums.size(); i++) {
		TeachChoose >> Nums[i];
	}
	TeachChoose.close();


	// Обучение сети
	long int k = 167; // Количество обучений нейросети
	double summ; // Переменная суммы
	double y; // Переменная выхода сети

	for (long int i = 0; i < k; i++) {
		Teacher.shuffle(nums, 10); // Тасование последовательности
		for (int j = 0; j < 10; j++) {
			for (int l = 0; l < 10; l++) {
				summ = Neyron.Summator(Nums[nums[j]], W[l]); // Получение взвешенной суммы
				y = Neyron.FunkActiv(summ, F); // Получение ответа нейрона
				if (nums[j] != l) { // Если текущии веса не совпадают с поданной на вход цифрой то ожидаемый ответ сети -1
					Teacher.WTSimplePerceptron(-1, y, W[l], Nums[nums[j]]);
				}
				else { // Если совпадают, то ожидаемый ответ сети 1
					Teacher.WTSimplePerceptron(1, y, W[l], Nums[nums[j]]);
				}
			}
		}
	}

		// Сохраняем веса
		ofstream fWeights;
		fWeights.open("./resources/Weights.txt");
		for (int i = 0; i < 10; i++) {
			fWeights << W[i];
		}
		fWeights.close();

#else
	// Считывание весов
	ifstream fWeights;
	fWeights.open("Weights.txt");
	for (int i = 0; i < 10; i++) {
		fWeights >> W[i];
	}
	fWeights.close();
#endif // Teach

	// Создание тестовой выборки
	vector<Matrix<double>> Tests(20);

	// Считывание тестовой выборки из файла
	ifstream Testsnums;
	Testsnums.open("./resources/Tests.txt");
	for (int i = 0; i < 20; i++) {
		Testsnums >> Tests[i];
	}

	// Вывод на экран реультатов тестирования сети
	cout << "Test network:" << endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			if (Neyron.FunkActiv(Neyron.Summator(Tests[i], W[j]), F) == 1)
				// Вывод результатов на экран
				cout << "Test " << i << " : " << "recognized " << j << endl;
		}
	}

	// Вывод на экран реультатов тестирования сети
	cout << "Test resilience:" << endl;
	for (int i = 10; i < 20; i++) {
		for (int j = 0; j < 10; j++) {
			if (Neyron.FunkActiv(Neyron.Summator(Tests[i], W[j]), F) == 1)
				// Вывод результатов на экран
				cout << "Test " << i << " : " << "recognized " << j << endl;
		}
	}

	// Вывод весов сети
	cout << endl << "Weights of network: " << endl;
	for (int i = 0; i < 10; i++) {
		cout << "Weight " << i << "-th neyron's:" << endl;
		W[i].Out();
		cout << endl;
	}

	return 0;

}

// отношение числа проходов к кофиценту скорости обучения равно 166,6666666666666
