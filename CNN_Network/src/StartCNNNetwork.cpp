﻿//: Нейросеть распознающая все цифры

#include "DenceLayers.h"
#include "FlattenLayers.h"
#include "ConvolutionLayers.h"
#include "MaxpoolingLayers.h"
#include <vector>
#include <iostream>
#include <fstream>


using namespace std;
using namespace ANN;
// Макрос режима работы программы (с обучением или без)

#define Teach

double mean(const double* arr, size_t len){
    double result = accumulate(arr, arr+len, 0.0);

    return result/len;
}
// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;
using namespace ANN;
int main()
{
// Создание функции ошибки
    RMS_errorD<double> MM;

    // Создание метрики
    ClassificationAccuracy<double> caccur;
    Accuracy<double> accur;
    RMS_error<double> rms;

    // Создание градиентного спуска
    SGD_Momentum<double > G(0.09, 0.9);


	// Создание функтора
	Sigm<double> F_1(1);
	Relu<double> F_2(1);

	// Производная функтора
	SigmD<double> f_1(1);
	ReluD<double> f_2(1);

    // Создание инициализатора
    glorot_uniform<double> I1(25, 37);
    glorot_uniform<double> I2(54, 27);
    glorot_uniform<double> I3(300, 128);
    glorot_uniform<double> I4(128, 84);
    glorot_uniform<double> I5(84, 10);

	// Установка зерна для выдачи рандомных значений

	// Размер входной матрицы
	const int image_width = 28;
	const int image_height = 28;

	// Размер фильтров (ядер свертки)
	const int filter_width = 5;
	const int filter_height = 5;
	const int filter1_width = 5;
	const int filter1_height = 5;

	// Размер матрицы нейронов
	const int neyron_width = 100;
	const int neyron_height = 4;
	const int neyron1_width = 120;
	const int neyron1_height = 1;

	// Количество фильтров
	const int f1_count = 5;
	const int k = 5;
	const int f2_count = f1_count;

	// Создание слоев
    D_ConvolutionLayer conv1(6, 5,5,1, I1, 1);
    D_MaxpoolingLayer maxp1(2,2);

    D_ConvolutionLayer conv2(12, 3,3,6, I2, 1);
    D_MaxpoolingLayer maxp2(2,2);

    D_FlattenLayer flat1;

	D_DenceLayer dence1(128,300,F_2, f_2,I3, 0.0);
	D_DenceLayer dence2(84, 128,F_2,f_2,I4, 0.0);
	D_DenceLayer dence3(10, 84,F_1,f_1,I5, 0.0);

	// Матрица выхода сети
	Matrix<double> MATRIX_OUT_1(1, 64);
	Matrix<double> MATRIX_OUT_2(1, 10);
	Matrix<double> MATRIX_OUT_3(1, 10);

	Matrix<double> correct(1, 10);
	Matrix<double> metrix(3, 60000);
	Matrix<double> error(10, 10);

	string file;

	// Матрицы изображений
	// Матрица входного изображения
    D_Tensor IMAGE_1(28, 28,1);
	// Вектор матриц изображений после первого сверточного слоя
    D_Tensor IMAGE_2(24,24, f1_count);
	// Вектор матриц изображений после первого подвыборочного слоя
    D_Tensor IMAGE_3(12,12, f1_count);
	// Вектор матриц изображений после второго сверточного слоя
    D_Tensor IMAGE_4(10, 10, f2_count);
	// Вектор матриц изображений после второго подвыборочного слоя
    D_Tensor IMAGE_5(5, 5, f2_count);

	// Вектор, передающийся в перцептрон (состоит из всех карт последнего подвыборочного слоя)
	Matrix<double> IMAGE_OUT(neyron_height, neyron_width);

	// Переменная максимума
	int max = 0;

	// Переменная прогресс бара
	int procent = 0;

#ifdef Teach

	// Матрицы ошибок сверточной сети
	// Вектор матриц ошибок первого сверточного слоя
	D_Tensor IMAGE_2_D(24, 24, f1_count);
	// Вектор матриц ошибок первого подвыборочного слоя
    D_Tensor IMAGE_3_D(12, 12, f1_count);
	// Вектор матриц ошибок второго сверточного слоя
    D_Tensor IMAGE_4_D(10, 10, f2_count);
	// Вектор матриц ошибок второго подвыборочного слоя
    D_Tensor IMAGE_5_D(5,5 , f2_count);

	// Матрица ошибки выхода изображения
	Matrix<double> IMAGE_OUT_D(1, 300);

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	long int koll = 5000; // Количество обучений нейросети (по совместительству количество разных шрифтов)

	// Создание обучающей выборки
	vector< vector <Matrix<double> > > Nums(10);
	for (int i = 0; i < 10; i++) {
		Nums[i] = vector<Matrix<double> >(koll);
	}

    dence1.saveWeightsToFile("./resources/4Weights1.txt");
    dence2.saveWeightsToFile("./resources/4Weights2.txt");
    dence3.saveWeightsToFile("./resources/4Weights3.txt");
    conv1.saveFiltersToFile("./resources/4Filters1.txt");
    conv2.saveFiltersToFile("./resources/4Filters2.txt");
	// Считывание обучающей выборки
	string folder = "../Image_to_txt/resources/";
	string path;
	ifstream input;
	for (int i = 0; i < 10; i++) {
		file = to_string(i) + ".txt";
		path = folder + file;
		input.open(path);
		for (int j = 0; j < koll; j++) {
			input >> Nums[i][j];
		}
		input.close();
	}


//    dence1.getWeightsFromFile("./resources/Weights1.txt");
//    dence2.getWeightsFromFile("./resources/Weights2.txt");
//    dence3.getWeightsFromFile("./resources/Weights3.txt");
//    conv1.getFiltersFromFile("./resources/Filters1.txt");
//    conv2.getFiltersFromFile("./resources/Filters2.txt");

    int nums[10] = { 0,1,2,3,4,5,6,7,8,9 };
	// Обучение сети
	for(size_t epoch = 0; epoch < 5; epoch++) {
        for (long int i = 0; i < koll; i++) {
//            shuffle(nums, nums+10,  default_random_engine(time(0))); // Тасование последовательности
            for (int j = 0; j < 10; j++) { // Цикл прохода по обучающей выборке
                // Работа сети
                // Обнуление переменной максимума
                max = 0;
                // Считывание картика поданной на вход сети
                IMAGE_1[0] = Nums[j][i];
                // Проход картинки через первый сверточный слой
                IMAGE_2 = conv1.passThrough(IMAGE_1);
                // Операция макспулинга
                IMAGE_3 = maxp1.passThrough(IMAGE_2);
                // Проход картинки через второй сверточный слой
                IMAGE_4 = conv2.passThrough(IMAGE_3);
                // Операция макспулинга
                IMAGE_5 = maxp2.passThrough(IMAGE_4);
                // Переход со сверточных слоев к перцептрону
                IMAGE_OUT = flat1.passThrough(IMAGE_5);
                // Проход по перцептрону
                // Проход по первому слою
                MATRIX_OUT_1 = dence1.passThrough(IMAGE_OUT);
                // Проход по второму слою
                MATRIX_OUT_2 = dence2.passThrough(MATRIX_OUT_1);
                MATRIX_OUT_3 = dence3.passThrough(MATRIX_OUT_2);
                for (int l = 0; l < 10; l++) { // Получение результатов сети
                    if (l == j)
                        correct[0][l] = 1;
                    if (l != j)
                        correct[0][l] = 0;
                }
                //Расчет ошибки
                error = loss_function(MM, MATRIX_OUT_3, correct);
                metrix[0][i*10+j] = metric_function(accur, MATRIX_OUT_3, correct);
                metrix[1][i*10+j] = metric_function(caccur, MATRIX_OUT_3, correct);
                metrix[2][i*10+j] = metric_function(rms, MATRIX_OUT_3, correct);
                // Обучение сети
                dence3.BackPropagation(error);
                dence2.BackPropagation(dence3);
                dence1.BackPropagation(dence2);

                // Копирование ошибки на подвыборочный слой
                IMAGE_5_D = flat1.passBack(dence1, 5, 5, 12);

                // Распространение ошибки на сверточный слой
                IMAGE_4_D = maxp2.BackPropagation(IMAGE_4, IMAGE_5, IMAGE_5_D);

                // Распространение ошибки на подвыборочный слой
                IMAGE_3_D = conv2.BackPropagation(IMAGE_4_D);

                // Распространение ошибки на сверточный слой
                IMAGE_2_D = maxp1.BackPropagation(IMAGE_2, IMAGE_3, IMAGE_3_D);

                // Примемение градиентного спуска
                // Первый сверточный слой
                conv1.GradDes(G, IMAGE_1, IMAGE_2_D);
                // Второй сверточный слой
                conv2.GradDes(G, IMAGE_3, IMAGE_4_D);
                // Перцептрон
                // Первый слой
                dence1.GradDes(G, IMAGE_OUT);
                // Второй слой
                dence2.GradDes(G, MATRIX_OUT_1);
                dence3.GradDes(G, MATRIX_OUT_2);


                // Обнуление ошибок
                dence1.setZero();
                dence2.setZero();
                dence3.setZero();
                // Обнуления вектора ошибок
                IMAGE_OUT_D.Fill(0);
                // "Замедление обучения сети"
                cout << "||";
        }

//            error = loss_function(MM, output.getPodmatrix(0, 0, 1, 10),
//                    correct.getPodmatrix(0, 0, 1, 10));
//            for(size_t iii = 1; iii < 10; iii++ ){
//                error = error + loss_function(MM, output.getPodmatrix(iii, 0, 1, 10),
//                                              correct.getPodmatrix(iii, 0, 1, 10));
//            }
//            for(size_t l = 0; l < 10; l++){
//                error[0][l] /= 10;
//            }
//            // Обучение сети
//            dence3.BackPropagation(error);
//            dence2.BackPropagation(dence3);
//            dence1.BackPropagation(dence2);
//
//            // Копирование ошибки на подвыборочный слой
//            IMAGE_5_D = flat1.passBack(dence1, 1, 72, 5, 5);
//
//            // Распространение ошибки на сверточный слой
//            IMAGE_4_D = BackPropagation(IMAGE_4, IMAGE_5, IMAGE_5_D, 2, 2);
//
//            // Распространение ошибки на подвыборочный слой
//            IMAGE_3_D = BackPropagation(IMAGE_4_D, conv2, 1);
//
//            // Распространение ошибки на сверточный слой
//            IMAGE_2_D = BackPropagation(IMAGE_2, IMAGE_3, IMAGE_3_D, 2, 2);
//
//            // Примемение градиентного спуска
//            // Первый сверточный слой
//            conv1.GradDes(G, IMAGE_1, IMAGE_2_D);
//            // Второй сверточный слой
//            conv2.GradDes(G, IMAGE_3, IMAGE_4_D);
//            // Перцептрон
//            // Первый слой
//            dence1.GradDes(G, IMAGE_OUT);
//            // Второй слой
//            dence2.GradDes(G, MATRIX_OUT_1);
//            dence3.GradDes(G, MATRIX_OUT_2);
//
//
//            // Обнуление ошибок
//            dence1.setZero();
//            dence2.setZero();
//            dence3.setZero();
//            // Обнуления вектора ошибок
//            IMAGE_OUT_D.Fill(0);
            if(epoch == 0) {
                cout << "] accuracy: ";
                cout << setw(5) << setprecision(4) << left << mean(metrix[0], i * 10 + 10);
                cout << " class_accuracy: ";
                cout << setw(3) << setprecision(4) << left << mean(metrix[1], i * 10 + 10);
                cout << " loss: " << setw(5) << setprecision(4) << left << mean(metrix[2], i * 10 + 10) << endl;
            }else{
                cout << "] accuracy: ";
                cout << setw(5) << setprecision(4) << left << mean(metrix[0], koll*10);
                cout << " class_accuracy: ";
                cout << setw(3) << setprecision(4) << left << mean(metrix[1], i * 10 + 10);
                cout << " loss: " << setw(5) << setprecision(4) << left << mean(metrix[2], koll*10) << endl;
            }

            if(i % 1000 == 0){
				// Сохранение весов
				dence1.saveWeightsToFile("./resources/Weights1.txt");
				dence2.saveWeightsToFile("./resources/Weights2.txt");
				dence3.saveWeightsToFile("./resources/Weights3.txt");
				conv1.saveFiltersToFile("./resources/Filters1.txt");
				conv2.saveFiltersToFile("./resources/Filters2.txt");

            }
        }
        cout << "epoch: " << epoch << endl;
    }
	cout << "end" << endl;

	// Сохранение весов
    dence1.saveWeightsToFile("./resources/Weights1.txt");
    dence2.saveWeightsToFile("./resources/Weights2.txt");
    dence3.saveWeightsToFile("./resources/Weights3.txt");
    conv1.saveFiltersToFile("./resources/Filters1.txt");
    conv2.saveFiltersToFile("./resources/Filters2.txt");

#else
    dence1.getWeightsFromFile("./resources/Weights1.txt");
    dence2.getWeightsFromFile("./resources/Weights2.txt");
    dence3.getWeightsFromFile("./resources/Weights3.txt");
    conv1.getFiltersFromFile("./resources/Filters1.txt");
    conv2.getFiltersFromFile("./resources/Filters2.txt");

#endif // Teach

	 // Создание тестовой выборки
	 Matrix<Matrix<Matrix<double> > > TestNums(1, 10);
	 for (int i = 0; i < 10; i++) {
		 TestNums[0][i] = Matrix<Matrix<double> >(1, 5000);
	 }
	 // Считывание тестовой выборки
	 for (int i = 0; i < 10; i++) {
		 file = to_string(i) + ".txt";
		 getDataFromTextFile(TestNums[0][i], "../Image_to_txt/resources/"+file);
	 }
	// Переменная ошибок сети
	int errors_network = 0;
	// Вывод на экран реультатов тестирования сети
	cout << "Test network:" << endl;
	for (int i = 0; i < 10; i++) { // Цикл прохода по тестовой выборке
		for (int j = 0; j < 900; j++) {
            // Работа сети
            // Обнуление переменной максимума
            max = 0;
            // Считывание картика поданной на вход сети
            IMAGE_1[0] = TestNums[0][i][0][j];
            // Проход картинки через первый сверточный слой
            IMAGE_2 = conv1.passThrough(IMAGE_1);
            // Операция макспулинга
            IMAGE_3 = maxp1.passThrough(IMAGE_2);
            // Проход картинки через второй сверточный слой
            IMAGE_4 = conv2.passThrough(IMAGE_3);
            // Операция макспулинга
            IMAGE_5 = maxp2.passThrough(IMAGE_4);
            // Переход со сверточных слоев к перцептрону
            IMAGE_OUT = flat1.passThrough(IMAGE_5);
            // Проход по перцептрону
            // Проход по первому слою
            MATRIX_OUT_1 = dence1.passThrough(IMAGE_OUT);
            // Проход по второму слою
            MATRIX_OUT_2 = dence2.passThrough(MATRIX_OUT_1);
            MATRIX_OUT_3 = dence3.passThrough(MATRIX_OUT_2);
			max = std::max_element(MATRIX_OUT_3[0], MATRIX_OUT_3[0]+10) - MATRIX_OUT_3[0];
			// Вывод результатов на экран
//			cout << "Test " << i << " : " << "recognized " << max << ' ' << MATRIX_OUT_3[0][max] << endl;
			// Подсчет ошибок
			if (max != i) {
				errors_network++;
			}
		}
	}
	// Вывод количества ошибок на экран
	cout << errors_network << endl;

	// Считывание тестовой выборки
	for (int i = 0; i < 10; i++) {
		file = to_string(i) + "_tests.txt";
		getDataFromTextFile(TestNums[0][i], "../Image_to_txt/resources/"+file);
	}
	// Переменная количества ошибок на тестовой выборке
	int errors_resilience = 0;
	// Вывод на экран реультатов тестирования сети
	cout << "Test resilience:" << endl;
	for (int i = 0; i < 10; i++) { // Цикл прохода по тестовой выборке
		for (int j = 0; j < 800; j++) {
            // Работа сети
            // Обнуление переменной максимума
            max = 0;
            // Считывание картика поданной на вход сети
            IMAGE_1[0] = TestNums[0][i][0][j];
            // Проход картинки через первый сверточный слой
            IMAGE_2 = conv1.passThrough(IMAGE_1);
            // Операция макспулинга
            IMAGE_3 = maxp1.passThrough(IMAGE_2);
            // Проход картинки через второй сверточный слой
            IMAGE_4 = conv2.passThrough(IMAGE_3);
            // Операция макспулинга
            IMAGE_5 = maxp2.passThrough(IMAGE_4);
            // Переход со сверточных слоев к перцептрону
            IMAGE_OUT = flat1.passThrough(IMAGE_5);
            // Проход по перцептрону
            // Проход по первому слою
            MATRIX_OUT_1 = dence1.passThrough(IMAGE_OUT);
            // Проход по второму слою
            MATRIX_OUT_2 = dence2.passThrough(MATRIX_OUT_1);
            MATRIX_OUT_3 = dence3.passThrough(MATRIX_OUT_2);
            max = std::max_element(MATRIX_OUT_3[0], MATRIX_OUT_3[0]+10) - MATRIX_OUT_3[0];
            // Вывод результатов на экран
//            cout << "Test " << i << " : " << "recognized " << max << ' ' << MATRIX_OUT_3[0][max] << endl;
            // Подсчет ошибок
            if (max != i) {
				errors_resilience++;
            }
		}
	}
	// Вывод на экран реультатов тестирования сети
	cout << errors_resilience << endl;
	return 0;

}