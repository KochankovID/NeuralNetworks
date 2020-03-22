//: Нейросеть распознающая все цифры

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

// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;
using namespace ANN;
int main()
{
// Создание функции ошибки
    RMS_errorD<double> MM;

    // Создание метрики
    Accuracy<double> accur;
    RMS_error<double> rms;

    // Создание градиентного спуска
    SimpleGrad<double> G0(1);
    SimpleGrad<double> G1(0.1);
    SimpleGrad<double> G2(0.01);
    SimpleGrad<double> G3(0.0001);
    SimpleGrad<double> G4(0.0000001);

	// Создание функтора
	Sigm<double> F_1(0.06);

	// Производная функтора
	SigmD<double> f_1(0.06);

    // Создание инициализатора
    SimpleInitializator<double> I(1);
    allOne<double> I1(0);

	// Установка зерна для выдачи рандомных значений
	srand(time(0));

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
    D_ConvolutionLayer conv1(8, make_pair(5,5), I1, 1);
    D_MaxpoolingLayer maxp1(2,2);

    D_ConvolutionLayer conv2(16, make_pair(3,3), I1, 1);
    D_MaxpoolingLayer maxp2(2,2);

    D_FlattenLayer flat1;

	D_DenceLayer dence1(64,3200,F_1,f_1,I1);
	D_DenceLayer dence2(64,64,F_1,f_1,I1);
	D_DenceLayer dence3(10,64,F_1,f_1,I1);

	// Матрица выхода сети
	Matrix<double> MATRIX_OUT_1(1, 64);
	Matrix<double> MATRIX_OUT_2(1, 64);
	Matrix<double> MATRIX_OUT_3(1, 10);

	Matrix<double> output(10, 10);
	Matrix<double> correct(10, 10);
	Matrix<double> error(10, 10);


	// Матрицы изображений
	// Матрица входного изображения
    Matrix< Matrix<double> > IMAGE_1(1, 1);
	// Вектор матриц изображений после первого сверточного слоя
    Matrix< Matrix<double> > IMAGE_2(1, f1_count);
	// Вектор матриц изображений после первого подвыборочного слоя
    Matrix< Matrix<double> > IMAGE_3(1, f1_count);
	// Вектор матриц изображений после второго сверточного слоя
    Matrix< Matrix<double> > IMAGE_4(1, f2_count*f1_count);
	// Вектор матриц изображений после второго подвыборочного слоя
    Matrix< Matrix<double> > IMAGE_5(1, f2_count*f1_count);

	// Вектор, передающийся в перцептрон (состоит из всех карт последнего подвыборочного слоя)
	Matrix<double> IMAGE_OUT(neyron_height, neyron_width);

	// Переменная максимума
	int max = 0;

	// Переменная прогресс бара
	int procent = 0;

#ifdef Teach

	// Матрицы ошибок сверточной сети
	// Вектор матриц ошибок первого сверточного слоя
	Matrix< Matrix<double> > IMAGE_2_D(1, f1_count);
	// Вектор матриц ошибок первого подвыборочного слоя
    Matrix< Matrix<double> > IMAGE_3_D(1, f1_count);
	// Вектор матриц ошибок второго сверточного слоя
    Matrix< Matrix<double> > IMAGE_4_D(1, f2_count*f1_count);
	// Вектор матриц ошибок второго подвыборочного слоя
    Matrix< Matrix<double> > IMAGE_5_D(1, f2_count*f1_count);

	// Матрица ошибки выхода изображения
	Matrix<double> IMAGE_OUT_D(1, 400);

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	long int koll = 1000; // Количество обучений нейросети (по совместительству количество разных шрифтов)

	// Создание обучающей выборки
	vector< vector <Matrix<double> > > Nums(10);
	for (int i = 0; i < 10; i++) {
		Nums[i] = vector<Matrix<double> >(koll);
	}

	// Считывание весов
	// Опционально, используется для обучения
	/*ifstream fWeightss;
	fWeightss.open("./resources/Weights.txt");
	for (int i = 0; i < f1_count; i++) {
		fWeightss >> FILTERS[i];
	}
	for (int i = 0; i < f2_count; i++) {
		fWeightss >> FILTERS1[i];
	}
	fWeightss >> WEIGHTS;
	fWeightss >> WEIGHTS1;
	fWeightss.close();*/

	// Считывание обучающей выборки
	string folder = "../Image_to_txt/resources/";
	string file;
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

	// Обучение сети
	for(size_t epoch = 0; epoch < 1; epoch++) {
        for (long int i = 0; i < koll; i++) {
            for (int j = 0; j < 10; j++) { // Цикл прохода по обучающей выборке
                // Работа сети
                // Обнуление переменной максимума
                max = 0;
                // Считывание картика поданной на вход сети
                IMAGE_1[0][0] = Nums[j][i];
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
                        correct[j][l] = 1;
                    if (l != j)
                        correct[j][l] = 0;
                    output[j][l] = MATRIX_OUT_3[0][l];
                }
                // Расчет ошибки
                error = loss_function(MM, output.getPodmatrix(j, 0, 1, 10), correct.getPodmatrix(j, 0, 1, 10));

                // Обучение сети
                dence3.BackPropagation(error);
                dence2.BackPropagation(dence3);
                dence1.BackPropagation(dence2);

                // Копирование ошибки на подвыборочный слой
                IMAGE_5_D = flat1.passBack(dence1, 1, 128, 5, 5);

                // Распространение ошибки на сверточный слой
                IMAGE_4_D = BackPropagation(IMAGE_4, IMAGE_5, IMAGE_5_D, 2, 2);

                // Распространение ошибки на подвыборочный слой
                IMAGE_3_D = BackPropagation(IMAGE_4_D, conv2, 1);

                // Распространение ошибки на сверточный слой
                IMAGE_2_D = BackPropagation(IMAGE_2, IMAGE_3, IMAGE_3_D, 2, 2);

                // Примемение градиентного спуска
                // Первый сверточный слой
                conv1.GradDes(G4, IMAGE_1, IMAGE_2_D);
                // Второй сверточный слой
                conv2.GradDes(G3, IMAGE_3, IMAGE_4_D);
                // Перцептрон
                // Первый слой
                dence1.GradDes(G2, IMAGE_OUT);
                // Второй слой
                dence2.GradDes(G1, MATRIX_OUT_1);
                dence3.GradDes(G0, MATRIX_OUT_2);

                // Обнуление ошибок
                dence1.setZero();
                dence2.setZero();
                dence3.setZero();
                // Обнуления вектора ошибок
                IMAGE_OUT_D.Fill(0);
                // "Замедление обучения сети"
                cout << "||";
            }
            cout << "] accuracy: ";
            cout << metric_function(accur, output, correct);
            cout << " loss: " << metric_function(rms, output, correct) << endl;
        }
        cout << "epoch: " << epoch << endl;
    }

	// Сохранение весов
    dence1.saveWeightsToFile("./resources/Weights1.txt");
    dence2.saveWeightsToFile("./resources/Weights2.txt");
    conv1.saveFiltersToFile("./resources/Filters1.txt");
    conv2.saveFiltersToFile("./resources/Filters2.txt");

#else
	 //Считывание весов
	 ifstream fWeights;
	 fWeights.open("./resources/Weights.txt");
	 for (int i = 0; i < f1_count; i++) {
		 fWeights >> FILTERS[i];
	 }
	 for (int i = 0; i < f2_count; i++) {
		 fWeights >> FILTERS1[i];
	 }
	 fWeights >> WEIGHTS;
	 fWeights >> WEIGHTS1;
	 fWeights.close();
	 string folder;
	 string file;
	 string path;

#endif // Teach

	 // Создание тестовой выборки
	 Matrix<Matrix<Matrix<double> > > TestNums(1, 10);
	 for (int i = 0; i < 10; i++) {
		 TestNums[0][i] = Matrix<Matrix<double> >(1, 100);
	 }
	 // Считывание тестовой выборки
	 folder = "../Image_to_txt/resources/";
	 for (int i = 0; i < 10; i++) {
		 file = to_string(i) + ".txt";
		 path = folder + file;
		 ifstream inputt(path);
		 for (int j = 0; j < 30; j++) {
			 inputt >> TestNums[0][i][0][j];
		 }
		 inputt.close();
	 }
	// Переменная ошибок сети
	int errors_network = 0;
	// Вывод на экран реультатов тестирования сети
	cout << "Test network:" << endl;
	for (int i = 0; i < 10; i++) { // Цикл прохода по тестовой выборке
		for (int j = 0; j < 3; j++) {
            // Работа сети
            // Обнуление переменной максимума
            max = 0;
            // Считывание картика поданной на вход сети
            IMAGE_1[0][0] = Nums[j][i];
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
			cout << "Test " << i << " : " << "recognized " << max << ' ' << MATRIX_OUT_3[0][max] << endl;
			// Подсчет ошибок
			if (max != i) {
				errors_network++;
			}
		}
	}
	// Вывод количества ошибок на экран
	cout << errors_network << endl;

	// Считывание тестовой выборки
	folder = "../Image_to_txt/resources/";
	for (int i = 0; i < 10; i++) {
		file = to_string(i) + "_tests.txt";
		path = folder + file;
		ifstream inputt(path);
		for (int j = 0; j < 100; j++) {
			inputt >> TestNums[0][i][0][j];
		}
		inputt.close();
	}
	// Переменная количества ошибок на тестовой выборке
	int errors_resilience = 0;
	// Вывод на экран реультатов тестирования сети
	cout << "Test resilience:" << endl;
	for (int i = 0; i < 10; i++) { // Цикл прохода по тестовой выборке
		for (int j = 0; j < 100; j++) {
            // Работа сети
            // Обнуление переменной максимума
            max = 0;
            // Считывание картика поданной на вход сети
            IMAGE_1[0][0] = Nums[j][i];
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
            cout << "Test " << i << " : " << "recognized " << max << ' ' << MATRIX_OUT_3[0][max] << endl;
            // Подсчет ошибок
            if (max != i) {
                errors_network++;
            }
		}
	}
	// Вывод на экран реультатов тестирования сети
	cout << errors_resilience << endl;
	return 0;

}