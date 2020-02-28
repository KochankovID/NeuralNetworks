//: Нейросеть распознающая все цифры

#include "Perceptrons.h"
#include <vector>
#include <iostream>
#include <fstream>

// Макрос режима работы программы (с обучением или без)
#define Teach

// Сигмоида
class Sigm : public D_Func
{
public:
	Sigm(const double& a_) : D_Func(), a(a_) {};
	double a;
	double operator()(const double& x) {
		double f = 1;
		const double e = 2.7182818284;
		for (int i = 0; i < a*x; i++)
		{
			f *= 1 / e;
		}
		f++;
		return 1 / f;
	}
	~Sigm() {};
};

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

int main()
{
	// Создание перцептрона
	D_Perceptron Neyron;

	// Создание обучателя сети
	D_Leaning Teacher;
	Teacher.getE() = 1;

	// Создание функтора
	Sigm F(5.7);

	// Создание весов нейросети
	vector<Weights<double>> W(10);
	for (int i = 0; i < 10; i++) {
		W[i] = Weights<double>(5, 3);
	}

	double summ; // Переменная суммы
	int y; // Переменная выхода сети
	double Outs[10] = { 0,1,2,0 };// Массив выходов сети
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

	for (long int i = 0; i < k; i++) {
		Teacher.shuffle(nums, 10); // Тасование последовательности
		for (int j = 0; j < 10; j++) { // Проход по обучающей выборке
			for (int l = 0; l < 10; l++) { // Проход по всем нейронам сети
				summ = Neyron.Summator(Nums[nums[j]], W[l]); // Получение взвешенной суммы
				y = Neyron.FunkActiv(summ, F); // Получение ответа нейрона
				if (nums[j] != l) {
					// Если номер текущего нейрона не совпадает с текущей цифрой, то ожидаемый ответ 0
					Teacher.WTSimplePerceptron(0, y, W[l], Nums[nums[j]]);
				}
				else {
					// Если номер текущего нейрона совпадает с текущей цифрой, то ожидаемый ответ 1
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
	fWeights.open("./resources/Weights.txt");
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

	// Вывод на экран реультатов тестирования сети на обучающей выборке
	cout << "Test network:" << endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			summ = Neyron.Summator(Tests[i], W[j]); // Получение взвешенной суммы
			Outs[j] = Neyron.FunkActiv(summ, F);
		}
		y = max(Outs, 10);
		// Вывод результатов на экран
		cout << "Test " << i << " : " << "recognized " << y << endl;
	}

	// Вывод на экран реультатов тестирования сети на тестовой выборке
	cout << "Test resilience:" << endl;
	for (int i = 10; i < 20; i++) {
		for (int j = 0; j < 10; j++) {
			summ = Neyron.Summator(Tests[i], W[j]); // Получение взвешенной суммы
			Outs[j] = Neyron.FunkActiv(summ, F);
		}
		y = max(Outs, 10);
		// Вывод результатов на экран
		cout << "Test " << i << " : " << "recognized " << y << endl;
	}

	// Вывод весов сети на экран
	cout << endl << "Weights of network: " << endl;
	for (int i = 0; i < 10; i++) {
		cout << "Weight " << i << "-th neyron's:" << endl;
		W[i].Out();
		cout << endl;
	}

	return 0;

}