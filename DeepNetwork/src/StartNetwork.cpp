//: Нейросеть распознающая все цифры

#include "Perceptrons.h"
#include <vector>
#include <iostream>
#include <fstream>

// Макрос режима работы программы (с обучением или без)
#define Teach

// Улучшение читабильности программы
#define NUMBER nums[j]

// функтор
// Сигмоида
class Sigm : public DD_Func
{
public:
	Sigm(const double& a_) : DD_Func(), a(a_) {};
	double a;
	double operator()(const double& x) {
		double f = 1;
		const double e = 2.7182818284;
		if (x >= 0) {
			for (int i = 0; i < a*x; i++)
			{
				f *= 1 / e;
			}
		}
		else {
			for (int i = 0; i < abs(int(a*x)); i++)
			{
				f *= e;
			}
		}
		f++;
		return 1 / f;
	}
	~Sigm() {};
};

// Производная сигмоиды
class SigmD : public Sigm
{
public:
	SigmD(const double& a_) : Sigm(a_) {};
	double operator()(const double& x) {
		double f = 1;
		f = Sigm::operator()(x)*(1 - Sigm::operator()(x));
		return f;
	}
	~SigmD() {};
};

using namespace std;

int main()
{
	// Создание перцептрона
	DD_Perceptron Neyron;

	// Создание обучателя сети
	DD_Leaning Teacher;
	Teacher.getE() = 0.09;

	// Создание функтора
	Sigm F(3.4);

	// Создание производной функтора
	SigmD f(3.4);

	// Установка зерна для выдачи рандомных значений
	srand(time(0));

	// Количество нейронов первого слоя нейросети
	const int w1_count = 100;

	// Создание весов нейросети
	Matrix<Weights<double>> W(1, w1_count);
	for (int i = 0; i < w1_count; i++) {
		W[0][i] = Weights<double>(5, 3);
		for (int j = 0; j < W[0][i].getN(); j++) {
			for (int p = 0; p < W[0][i].getM(); p++) {
				W[0][i][j][p] = (p % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
			}
		}
		W[0][i].GetWBias() = (i % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
	}

	// Создания весов для второго слоя сети
	Matrix<Weights<double>> W1(1, 10);
	for (int i = 0; i < 10; i++) {
		W1[0][i] = Weights<double>(1, w1_count);
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
	double y[w1_count]; // Переменная выхода сети

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
	long int k = 70; // Количество обучений нейросети

	for (long int i = 1; i < k; i++) {
		Teacher.shuffle(nums, 10); // Тасование последовательности
		cout << i << endl;
		for (int j = 0; j < 10; j++) { // Цикл прохода по обучающей выборке
			for (int u = 0; u < 3; u++) {
				for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
					summ = Neyron.Summator(Nums[NUMBER], W[0][l]); // Получение взвешенной суммы
					m[0][l] = Neyron.FunkActiv(summ, F);
				}
				for (int l = 0; l < 10; l++) { // Цикл прохода по сети
					summ = Neyron.Summator(m, W1[0][l]); // Получение взвешенной суммы
					y[l] = Neyron.FunkActiv(summ, F); // Запись выхода l-того нейрона в массив выходов сети
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