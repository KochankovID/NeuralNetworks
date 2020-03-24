//: Нейросеть распознающая все цифры

#include "Neyrons.h"
#include "Data.h"
#include <vector>
#include <iostream>
#include "DenceLayer.h"
#include "Initializers.h"

// Макрос режима работы программы (с обучением или без)
#define Teach

// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;
using namespace ANN;

int main()
{
	// Создание функтора
	Sigm<double > F(1);
	Sigm<double > F1(1);

    RMS_error<double> MM;
    Accuracy<double> M;
	RMS_errorD<double> R;

	SimpleInitializatorPositive<double > I(1);

	// Создание производной функтора
	SigmD<double > FD(1);
	SigmD<double > FD1(1);

	// Установка зерна для выдачи рандомных значений
    // Количество нейронов первого слоя нейросети
	const int w1_count = 50;

    Matrix<double> output(10, 10);
    Matrix<double> correct(10, 10);

	// Создание весов нейросети
	DenceLayer<double> layer1(w1_count, 15, F, FD, I);


	// Создания весов для второго слоя сети
    DenceLayer<double> layer2(10, w1_count, F1, FD1, I);

	// Массив выходов первого слоя сети
	Matrix<double> matrix_out_1(1, w1_count);
	Matrix<double> matrix_out_2(1, w1_count);

	double summ; // Переменная суммы

#ifdef Teach

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	int nums[10] = { 0,1,2,3,4,5,6,7,8,9 };

	SimpleGrad<double> G(1);
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
		    matrix_out_1 = layer1.passThrough(Nums[0][NUMBER]);
            matrix_out_2 = layer2.passThrough(matrix_out_1);
				for (int l = 0; l < 10; l++) { // Цикл прохода по сети
                    output[j][l] = matrix_out_2[0][l];
                    if (l == NUMBER) {
                        correct[j][l] = 1;
                    }else{
                        correct[j][l] = 0;
                    }

				}

                error_vect = loss_function(R, output.getPodmatrix(j,0,1,10), correct.getPodmatrix(j,0,1,10));

				layer2.BackPropagation(error_vect);
                layer1.BackPropagation(layer2);

                layer1.GradDes(G, Nums[0][NUMBER]);
                layer2.GradDes(G, matrix_out_1);
                layer1.setZero();
                layer2.setZero();

			cout << "||";
		}
        cout << "] accuracy: ";
        cout << metric_function(M, output, correct);
        cout << " loss: " << metric_function(MM, output, correct) << endl;
	}

	// Сохраняем веса
	layer1.saveWeightsToFile("./resources/Weights0.txt");
    layer2.saveWeightsToFile("./resources/Weights1.txt");

#else
	// Считывание весов
	("./resources/Weights0.txt");
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
		matrix_out_1 = layer1.passThrough(Tests[0][j]);
		matrix_out_2 = layer2.passThrough(matrix_out_1);
        int max = max_element(matrix_out_2[0], matrix_out_2[0]+10) - matrix_out_2[0];
		// Вывод результатов на экран
		cout << "Test " << j << " : " << "recognized " << max << ' ' << matrix_out_2[0][max] << endl;
	}

	cout << "Test resilience:" << endl;
	for (int j = 10; j < 24; j++) { // Цикл прохода по тестовой выборке
        matrix_out_1 = layer1.passThrough(Tests[0][j]);
        matrix_out_2 = layer2.passThrough(matrix_out_1);
        int max = max_element(matrix_out_2[0], matrix_out_2[0]+10) - matrix_out_2[0];
        // Вывод результатов на экран
        cout << "Test " << j << " : " << "recognized " << max << ' ' << matrix_out_2[0][max] << endl;

	}

	// Вывод весов сети на экран
//	cout << endl << "Weights of network. First layer: " << endl;
//	for (int i = 0; i < 10; i++) {
//		cout << "Weight " << i << "-th neyron's:" << endl;
//		cout << layer1;
//		cout << endl;
//	}
//	cout << endl << "Weights of network. Second layer: " << endl;
//	for (int i = 0; i < 10; i++) {
//		cout << "Weight " << i << "-th neyron's:" << endl;
//		cout << layer2;
//		cout << endl;
//	}
	return 0;

}