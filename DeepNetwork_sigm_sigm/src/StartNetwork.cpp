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
	Sigm<double > F(0.9);

    RMS_error<double> MM;
    Accuracy<double> M;
	RMS_errorD<double> R;

	SimpleInitializator<double > I;
	// Создание производной функтора
	SigmD<double > FD(0.9);

	// Установка зерна для выдачи рандомных значений
    // Количество нейронов первого слоя нейросети
	const int w1_count = 50;

    Matrix<double> output(10, 10);
    Matrix<double> correct(10, 10);

	// Создание весов нейросети
	DenceLayer<double> layer1(w1_count, 15, F, FD, I);
//	Matrix<Neyron<double>> W(1, w1_count);
//	for (int i = 0; i < w1_count; i++) {
//		W[0][i] = Neyron<double>(5, 3);
//		for (int j = 0; j < W[0][i].getN(); j++) {
//			for (int p = 0; p < W[0][i].getM(); p++) {
//				W[0][i][j][p] = (p % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
//			}
//		}
//		W[0][i].GetWBias() = (i % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
//	}

	// Создания весов для второго слоя сети
    DenceLayer<double> layer2(10, w1_count, F, FD, I);
//	Matrix<Neyron<double>> W1(1, 10);
//	for (int i = 0; i < 10; i++) {
//		W1[0][i] = Neyron<double>(1, w1_count);
//		for (int j = 0; j < W1[0][i].getN(); j++) {
//			for (int p = 0; p < W1[0][i].getM(); p++) {
//				W1[0][i][j][p] = (p % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
//			}
//		}
//		W1[0][i].GetWBias() = (i % 2 ? ((double)rand() / RAND_MAX) : -((double)rand() / RAND_MAX));
//	}

	// Массив выходов первого слоя сети
	Matrix<double> matrix_out_1(1, w1_count);
	Matrix<double> matrix_out_2(1, w1_count);

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
		    matrix_out_1 = layer1.passThrough(Nums[0][NUMBER]);
//				for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
//					summ = W[0][l].Summator(Nums[0][NUMBER]); // Получение взвешенной суммы
//					m[0][l] = D_Neyron::FunkActiv(summ, F);
//				}
            matrix_out_2 = layer2.passThrough(matrix_out_1);
				for (int l = 0; l < 10; l++) { // Цикл прохода по сети
//					summ = W1[0][l].Summator(m); // Получение взвешенной суммы
//					output[j][l] = D_Neyron::FunkActiv(summ, F); // Запись выхода l-того нейрона в массив выходов сети
                    output[j][l] = matrix_out_2[0][l];
                    if (l == NUMBER) {
                        correct[j][l] = 1;
                    }else{
                        correct[j][l] = 0;
                    }

				}

                error_vect = loss_function(R, output.getPodmatrix(j,0,1,10), correct.getPodmatrix(j,0,1,10));

				cout << error_vect;

				layer2.BackPropagation(error_vect);
                layer1.BackPropagation(layer2);

                layer1.GradDes(G, Nums[0][NUMBER]);
                layer2.GradDes(G, matrix_out_1);
                layer1.setZero();
                layer2.setZero();

//				for (int l = 0; l < 10; l++) { // Распространение ошибки на скрытые слои нейросети
//					BackPropagation(W, W1[0][l]);
//				}
//				int dropout1 = rand()*10;
//				int dropout2 = rand()*10;
//				for (int l = 0; l < w1_count; l++) { // Примемение градиентного спуска по всем нейроннам первого слоя
//				    if((dropout1 == l)||(dropout2 == l)) continue;
//					GradDes(G, W[0][l], Nums[0][NUMBER], f);
//				}
//                dropout1 = rand()*10;
//                dropout2 = rand()*10;
//				for (int l = 0; l < 10; l++) { // Примемение градиентного спуска по всем нейроннам второго слоя
//                    if((dropout1 == l)||(dropout2 == l)) continue;
//					GradDes(G, W1[0][l], m, f);
//				}
//				for(int l = 0; l < 10; l++){
//				    W[0][l].GetD() = 0;
//				}
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
	cout << endl << "Weights of network. First layer: " << endl;
	for (int i = 0; i < 10; i++) {
		cout << "Weight " << i << "-th neyron's:" << endl;
		cout << layer1;
		cout << endl;
	}
	cout << endl << "Weights of network. Second layer: " << endl;
	for (int i = 0; i < 10; i++) {
		cout << "Weight " << i << "-th neyron's:" << endl;
		cout << layer2;
		cout << endl;
	}
	return 0;

}