//: Нейросеть распознающая все цифры

#include "DenceLayers.h"
#include "FlattenLayer.h"
#include "Filters.h"
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
    Accuracy<double> M;
    RMS_error<double> MMM;

    SimpleGrad<double> G(1);

	// Создание функтора
	Sigm<double> F_1(1);
    Relu<double> F_2(0.5);

	// Производная функтора
	SigmD<double> f_1(1);
    ReluD<double> f_2(0.5);

    // Создание инициализатора
    SimpleInitializatorPositive<double> I(2);
    SimpleInitializator<double> I1;

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

	// Количество нейронов

    FlattenLayer<double > layer3;
	D_DenceLayer layer1(100,400,F_1,f_1,I1);
	D_DenceLayer layer2(10,100,F_1,f_1,I1);

	const int w1_count = 120;
	const int w2_count = 10;

	// Кофицент создания весов
	const double decade = 0.1;

	// Создание весов фильтров первого слоя
	Matrix<Filter<double> > FILTERS(1,f1_count);
	for (int i = 0; i < f1_count; i++) {
		FILTERS[0][i] = Filter<double>(filter_height, filter_width);
		for (int j = 0; j < FILTERS[0][i].getN(); j++) {
			for (int p = 0; p < FILTERS[0][i].getM(); p++) {
				FILTERS[0][i][j][p] = (p % 2 ? ((double)rand() / (RAND_MAX*decade)) : -((double)rand() / (RAND_MAX * decade)));
			}
		}
	}

	// Создание весов фильтров второго слоя
	Matrix<Filter<double> > FILTERS1(1, f2_count);
	for (int i = 0; i < f2_count; i++) {
		FILTERS1[0][i] = Filter<double>(filter1_height, filter1_width);
		for (int j = 0; j < FILTERS1[0][i].getN(); j++) {
			for (int p = 0; p < FILTERS1[0][i].getM(); p++) {
				FILTERS1[0][i][j][p] = (p % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
			}
		}
	}

//	// Создание весов перового слоя перцептрона
//	Matrix<Weights<double> > WEIGHTS(1, w1_count);
//	for (int i = 0; i < w1_count; i++) {
//		WEIGHTS[0][i] = Weights<double>(neyron_height, neyron_width);
//		for (int j = 0; j < WEIGHTS[0][i].getN(); j++) {
//			for (int p = 0; p < WEIGHTS[0][i].getM(); p++) {
//				WEIGHTS[0][i][j][p] = (p % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
//			}
//		}
//		WEIGHTS[0][i].GetWBias() = (i % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
//	}

//	// Создания весов для второго слоя перцептрона
//	Matrix<Weights<double> > WEIGHTS1(1, w2_count);
//	for (int i = 0; i < w2_count; i++) {
//		WEIGHTS1[0][i] = Weights<double>(neyron1_height, neyron1_width);
//		for (int j = 0; j < WEIGHTS1[0][i].getN(); j++) {
//			for (int p = 0; p < WEIGHTS1[0][i].getM(); p++) {
//				WEIGHTS1[0][i][j][p] = (p % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
//			}
//		}
//		WEIGHTS1[0][i].GetWBias() = (i % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
//	}

	// Матрица выхода сети
	Matrix<double> MATRIX_OUT_1(1, w1_count);
	Matrix<double> MATRIX_OUT_2(1, w1_count);
	Matrix<double> MATRIX_OUT_3(1, w2_count);
	Matrix<double> output(10, w2_count);
	Matrix<double> correct(10, w2_count);
	Matrix<double> error(10, w2_count);


    double summ; // Переменная суммы
    double y[w2_count]; // Переменная выхода сети

	// Матрицы изображений
	// Матрица входного изображения
    Matrix< Matrix<double> > IMAGE_1(1, f1_count);
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

	for(size_t i = 0; i < f2_count; i++){
        IMAGE_5_D[0][i] = Matrix<double>(4,4);
	}

	// Матрица ошибки выхода изображения
	Matrix<double> IMAGE_OUT_D(1, 400);
	IMAGE_OUT.Fill(0);

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	// Может как использоваться или не использоваться
	int nums[10] = { 0,1,2,3,4,5,6,7,8,9 };

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

	// Массив, нужный для подсчета ошибки
	double a[10];

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
	for (long int i = 0; i < koll; i++) {
		for (int j = 0; j < 10; j++) { // Цикл прохода по обучающей выборке
            // Работа сети
            // Обнуление переменной максимума
            max = 0;
            // Считывание картика поданной на вход сети
            for(size_t l = 0; l< f1_count; l++){
                IMAGE_1[0][l] = Nums[j][i];
            }
            // Проход картинки через первый сверточный слой
            for (int l = 0; l < f1_count; l++) {
                IMAGE_2[0][l] = FILTERS[0][l].Svertka(IMAGE_1[0][0], 1);
            }
            // Операция макспулинга
            for (int l = 0; l < f1_count; l++) {
                IMAGE_3[0][l] = Filter<double >::Pooling(IMAGE_2[0][l], 2, 2);
            }
            // Проход картинки через второй сверточный слой
            for (int l = 0; l < f1_count; l++) {
                for (int ll = 0; ll < f2_count; ll++) {
                    IMAGE_4[0][l*f2_count + ll] = FILTERS1[0][ll].Svertka(IMAGE_3[0][l],1);
                }
            }
            // Операция макспулинга
            for (int l = 0; l < f2_count*f1_count; l++) {
                IMAGE_5[0][l] = Filter<double >::Pooling(IMAGE_4[0][l], 2, 2);
            }

            IMAGE_OUT = layer3.passThrough(IMAGE_5);
            // Проход по перцептрону
            // Проход по первому слою

            MATRIX_OUT_1 = layer1.passThrough(IMAGE_OUT);
//				for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
//					summ = Neyron.Summator(IMAGE_OUT, WEIGHTS[0][l]); // Получение взвешенной суммы
//					MATRIX_OUT[0][l] = Neyron.FunkActiv(summ, F_2);
//				}
//				for (int l = 0; l < w2_count; l++) { // Цикл прохода по сети
//					summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]); // Получение взвешенной суммы
//					y[l] = Neyron.FunkActiv(summ, F_1); // Запись выхода l-того нейрона в массив выходов сети
//				}
            MATRIX_OUT_3 = layer2.passThrough(MATRIX_OUT_1);
            for (int l = 0; l < w2_count; l++) { // Получение результатов сети
                if (l == j)
                    correct[j][l] = 1;
                if (l != j)
                    correct[j][l] = 0;
                output[j][l] = MATRIX_OUT_3[0][l];
            }
            // Вывод распознанной цифры на экран для визуализации процесса обучения
            /*cout << max << ' ';*/
            // Расчет ошибки

            error = loss_function(MM, output.getPodmatrix(j,0,1,10), correct.getPodmatrix(j,0,1,10));
            // Вывод ошибки на экран
            /*cout << Teacher.RMS_error(a, y, w2_count) << endl;*/
            // Если ошибка мала, пропускаем цикл обучения, что бы избежать переобучения сети
//				if (Teacher.RMS_error(a, y, w2_count) < 0.3) {
//					continue;
//				}
            // Обучение сети
            layer2.BackPropagation(error);
            layer1.BackPropagation(layer2);

//				for (int l = 0; l < w2_count; l++) { // Расчет ошибки для выходного слоя
//					if (l == NUMBER) { // Если номер нейрона совпадает с поданной на вход цифрой, то ожидаеммый ответ 1
//						WEIGHTS1[0][l].GetD() = Teacher.PartDOutLay(1, y[l]); // Расчет ошибки
//					}
//					else {// Если номер нейрона совпадает с поданной на вход цифрой, то ожидаеммый ответ 1
//						WEIGHTS1[0][l].GetD() = Teacher.PartDOutLay(0, y[l]); // Расчет ошибки
//					}
//				}
//				// Распространение ошибки на скрытые слои перцептрона
//				for (int l = 0; l < w2_count; l++) {
//					Teacher.BackPropagation(WEIGHTS, WEIGHTS1[0][l]);
//				}
            // Распространение ошибки на выход картинки
//				for (int l = 0; l < w2_count; l++) {
//					TeacherCNN.Revers_Perceptron_to_CNN(IMAGE_OUT_D, WEIGHTS[0][l]);
//				}
            IMAGE_5_D = layer3.passBack(layer1, 1, 25, 4, 4);
            // Копирование ошибки на подвыборочный слой
//				for (int l = 0; l < f2_count; l++) {
//					for (int li = 0; li < 4; li++) {
//						for (int lj = 0; lj < 4; lj++) {
//							IMAGE_5_D[0][l][li][lj] = IMAGE_OUT_D[0][l * 4+ li*4 + lj];
//						}
//					}
//				}
            // Распространение ошибки на сверточный слой
            IMAGE_4_D = BackPropagation(IMAGE_4, IMAGE_5, IMAGE_5_D, 2, 2);
//            for (int l = 0; l < f2_count; l++) {
//                IMAGE_4_D[0][l] = BackPropagation(IMAGE_4[0][l], IMAGE_5[0][l], IMAGE_5_D[0][l], 2, 2);
//            }
            IMAGE_3_D = BackPropagation(IMAGE_4_D, FILTERS1, 1);
            // Распространение ошибки на подвыборочный слой
//            for (int l = 0; l < f1_count; l++) {
//                IMAGE_3_D[0][l] = BackPropagation(IMAGE_4_D[0][l*k], FILTERS1[0][l*k], 1);
//                for (int ll = 1; ll < k; ll++) {
//                    IMAGE_3_D[0][l] = IMAGE_3_D[0][l] + TeacherCNN.ReversConvolution(IMAGE_4_D[0][l*k + ll], FILTERS1[0][l*k + ll]);
//                }
//            }
            // Распространение ошибки на сверточный слой
            IMAGE_2_D = BackPropagation(IMAGE_1, IMAGE_2, IMAGE_3_D, 2, 2);

//            for (int l = 0; l < f1_count; l++) {
//                IMAGE_2_D[0][l] = TeacherCNN.ReversPooling(IMAGE_3_D[0][l], 2, 2);
//            }
            // Примемение градиентного спуска
            // Первый сверточный слой
            GradDes(G, IMAGE_1, IMAGE_2_D, FILTERS, 1);
//            for (int l = 0; l < f1_count; l++) {
//                GradDes(IMAGE_1, IMAGE_2_D[0][l], FILTERS[0][l]);
//            }
            // Второй сверточный слой
            GradDes(G, IMAGE_3, IMAGE_4_D, FILTERS1, 1);

//            for (int l = 0; l < f1_count; l++) {
//                for (int ll = 0; ll < k; ll++) {
//                    GradDes(IMAGE_3[0][l], IMAGE_4_D[0][l*k + ll], FILTERS1[0][l*k + ll]);
//                }
//            }
            // Перцептрон
            // Первый слой
//				for (int l = 0; l < w1_count; l++) { // Примемение градиентного спуска по всем нейроннам первого слоя
//					Teacher.GradDes(WEIGHTS[0][l], IMAGE_OUT, f_2, MATRIX_OUT[0][l]);
//				}
            layer1.GradDes(G,  IMAGE_OUT);
            layer2.GradDes(G, MATRIX_OUT_1);
            // Второй слой
//			for (int l = 0; l < w2_count; l++) { // Примемение градиентного спуска по всем нейроннам второго слоя
//					summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]);
//					Teacher.GradDes(WEIGHTS1[0][l], MATRIX_OUT, f_1, summ);
//				}
            // Обнуление ошибок
            layer1.setZero();
            layer2.setZero();
            // Обнуления вектора ошибок
            IMAGE_OUT_D.Fill(0);
            // "Замедление обучения сети"
//				Teacher.getE() -= Teacher.getE() * 0.0000001;
//				TeacherCNN.getE() -= TeacherCNN.getE() * 0.0000001;
            cout << "||";
        }
        cout << "] accuracy: ";
        cout << metric_function(M, output, correct);
        cout << " loss: " << metric_function(MMM, output, correct) << endl;
	}

	// Сохранение весов
    layer1.saveWeightsToFile("./resources/Weights1.txt");
    layer2.saveWeightsToFile("./resources/Weights2.txt");

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
			for(size_t l = 0; l< f1_count; l++){
				IMAGE_1[0][l] = TestNums[0][i][0][j];
			}
			// Проход картинки через первый сверточный слой
			for (int l = 0; l < f1_count; l++) {
				IMAGE_2[0][l] = FILTERS[0][l].Svertka(IMAGE_1[0][0], 1);
			}
			// Операция макспулинга
			for (int l = 0; l < f1_count; l++) {
				IMAGE_3[0][l] = Filter<double >::Pooling(IMAGE_2[0][l], 2, 2);
			}
			// Проход картинки через второй сверточный слой
			for (int l = 0; l < f1_count; l++) {
				for (int ll = 0; ll < f2_count; ll++) {
					IMAGE_4[0][l*f2_count + ll] = FILTERS1[0][ll].Svertka(IMAGE_3[0][l],1);
				}
			}
			// Операция макспулинга
			for (int l = 0; l < f2_count*f1_count; l++) {
				IMAGE_5[0][l] = Filter<double >::Pooling(IMAGE_4[0][l], 2, 2);
			}

			IMAGE_OUT = layer3.passThrough(IMAGE_5);
			// Проход по перцептрону
			// Проход по первому слою

			MATRIX_OUT_1 = layer1.passThrough(IMAGE_OUT);
//				for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
//					summ = Neyron.Summator(IMAGE_OUT, WEIGHTS[0][l]); // Получение взвешенной суммы
//					MATRIX_OUT[0][l] = Neyron.FunkActiv(summ, F_2);
//				}
//				for (int l = 0; l < w2_count; l++) { // Цикл прохода по сети
//					summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]); // Получение взвешенной суммы
//					y[l] = Neyron.FunkActiv(summ, F_1); // Запись выхода l-того нейрона в массив выходов сети
//				}
			MATRIX_OUT_3 = layer2.passThrough(MATRIX_OUT_1);
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
			for(size_t l = 0; l< f1_count; l++){
				IMAGE_1[0][l] = TestNums[0][i][0][j];
			}
			// Проход картинки через первый сверточный слой
			for (int l = 0; l < f1_count; l++) {
				IMAGE_2[0][l] = FILTERS[0][l].Svertka(IMAGE_1[0][0], 1);
			}
			// Операция макспулинга
			for (int l = 0; l < f1_count; l++) {
				IMAGE_3[0][l] = Filter<double >::Pooling(IMAGE_2[0][l], 2, 2);
			}
			// Проход картинки через второй сверточный слой
			for (int l = 0; l < f1_count; l++) {
				for (int ll = 0; ll < f2_count; ll++) {
					IMAGE_4[0][l*f2_count + ll] = FILTERS1[0][ll].Svertka(IMAGE_3[0][l],1);
				}
			}
			// Операция макспулинга
			for (int l = 0; l < f2_count*f1_count; l++) {
				IMAGE_5[0][l] = Filter<double >::Pooling(IMAGE_4[0][l], 2, 2);
			}

			IMAGE_OUT = layer3.passThrough(IMAGE_5);
			// Проход по перцептрону
			// Проход по первому слою

			MATRIX_OUT_1 = layer1.passThrough(IMAGE_OUT);
//				for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
//					summ = Neyron.Summator(IMAGE_OUT, WEIGHTS[0][l]); // Получение взвешенной суммы
//					MATRIX_OUT[0][l] = Neyron.FunkActiv(summ, F_2);
//				}
//				for (int l = 0; l < w2_count; l++) { // Цикл прохода по сети
//					summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]); // Получение взвешенной суммы
//					y[l] = Neyron.FunkActiv(summ, F_1); // Запись выхода l-того нейрона в массив выходов сети
//				}
			MATRIX_OUT_3 = layer2.passThrough(MATRIX_OUT_1);
//			for (int l = 0; l < w2_count; l++) { // Цикл прохода по сети
//				summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]); // Получение взвешенной суммы
//				y[l] = Neyron.FunkActiv(summ, F_2); // Запись выхода l-того нейрона в массив выходов сети
//			}
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