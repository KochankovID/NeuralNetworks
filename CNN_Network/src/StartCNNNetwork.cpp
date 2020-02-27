//: Нейросеть распознающая все цифры

#include "Perceptrons.h"
#include "CNNs.h"
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
// Макрос режима работы программы (с обучением или без)

#define Teach

// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;

int main()
{
	// Создание перцептрона
	D_Perceptron Neyron;

	// Создание обучателя сети
	D_Leaning Teacher;
	Teacher.getE() = 0.00064;

	// Создание CNN
	D_NeyronCnn NeyronCNN;

	// Создание обучателя CNN сети
	D_CNNLeaning TeacherCNN;
	TeacherCNN.getE() = 0.000006;

	// Создание функтора
	Sigm<double> F_1(1);
    Relu<double> F_2(1);

	// Производная функтора
	SigmD<double> f_1(1);
    ReluD<double> f_2(1);

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
	const int f2_count = f1_count * k;

	// Количество нейронов
	const int w1_count = 120;
	const int w2_count = 10;
	
	// Кофицент создания весов
	const int decade = 0.1;

	// Создание весов фильтров первого слоя
	vector<Filter<double> > FILTERS(f1_count);
	for (int i = 0; i < f1_count; i++) {
		FILTERS[i] = Filter<double>(filter_height, filter_width);
		for (int j = 0; j < FILTERS[i].getN(); j++) {
			for (int p = 0; p < FILTERS[i].getM(); p++) {
				FILTERS[i][j][p] = (p % 2 ? ((double)rand() / (RAND_MAX*decade)) : -((double)rand() / (RAND_MAX * decade)));
			}
		}
	}

	// Создание весов фильтров второго слоя
	vector<Filter<double> > FILTERS1(f2_count);
	for (int i = 0; i < f2_count; i++) {
		FILTERS1[i] = Filter<double>(filter1_height, filter1_width);
		for (int j = 0; j < FILTERS1[i].getN(); j++) {
			for (int p = 0; p < FILTERS1[i].getM(); p++) {
				FILTERS1[i][j][p] = (p % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
			}
		}
	}

	// Создание весов перового слоя перцептрона
	Matrix<Weights<double> > WEIGHTS(1, w1_count);
	for (int i = 0; i < w1_count; i++) {
		WEIGHTS[0][i] = Weights<double>(neyron_height, neyron_width);
		for (int j = 0; j < WEIGHTS[0][i].getN(); j++) {
			for (int p = 0; p < WEIGHTS[0][i].getM(); p++) {
				WEIGHTS[0][i][j][p] = (p % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
			}
		}
		WEIGHTS[0][i].GetWBias() = (i % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
	}

	// Создания весов для второго слоя перцептрона
	Matrix<Weights<double> > WEIGHTS1(1, w2_count);
	for (int i = 0; i < w2_count; i++) {
		WEIGHTS1[0][i] = Weights<double>(neyron1_height, neyron1_width);
		for (int j = 0; j < WEIGHTS1[0][i].getN(); j++) {
			for (int p = 0; p < WEIGHTS1[0][i].getM(); p++) {
				WEIGHTS1[0][i][j][p] = (p % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
			}
		}
		WEIGHTS1[0][i].GetWBias() = (i % 2 ? ((double)rand() / (RAND_MAX * decade)) : -((double)rand() / (RAND_MAX * decade)));
	}

	// Матрица выхода сети
	Matrix<double> MATRIX_OUT(1, w1_count);

    double summ; // Переменная суммы
    double y[w2_count]; // Переменная выхода сети

	// Матрицы изображений
	// Матрица входного изображения
	Matrix<double> IMAGE_1(image_height, image_width);
	// Вектор матриц изображений после первого сверточного слоя
	vector< Matrix<double> > IMAGE_2(f1_count);
	// Вектор матриц изображений после первого подвыборочного слоя
	vector< Matrix<double> > IMAGE_3(f1_count);
	// Вектор матриц изображений после второго сверточного слоя
	vector< Matrix<double> > IMAGE_4(f2_count);
	// Вектор матриц изображений после второго подвыборочного слоя
	vector< Matrix<double> > IMAGE_5(f2_count);

	// Вектор, передающийся в перцептрон (состоит из всех карт последнего подвыборочного слоя)
	Matrix<double> IMAGE_OUT(neyron_height, neyron_width);

	// Переменная максимума
	int max = 0;

	// Переменная прогресс бара
	int procent = 0;

#ifdef Teach

	// Матрицы ошибок сверточной сети
	// Вектор матриц ошибок первого сверточного слоя
	vector< Matrix<double> > IMAGE_2_D(f1_count);
	// Вектор матриц ошибок первого подвыборочного слоя
	vector< Matrix<double> > IMAGE_3_D(f1_count);
	// Вектор матриц ошибок второго сверточного слоя
	vector< Matrix<double> > IMAGE_4_D(f2_count);
	// Вектор матриц ошибок второго подвыборочного слоя
	vector< Matrix<double> > IMAGE_5_D(f2_count);

	for(auto& mat : IMAGE_5_D){
	    mat = Matrix<double>(4,4);
	}

	// Матрица ошибки выхода изображения
	Matrix<double> IMAGE_OUT_D(neyron_height, neyron_width);
	IMAGE_OUT.Fill(0);

	// Последовательность цифр, тасуемая для получения равномерной рандомизации
	// Может как использоваться или не использоваться
	int nums[10] = { 0,1,2,3,4,5,6,7,8,9 };

	long int koll = 5000; // Количество обучений нейросети (по совместительству количество разных шрифтов)

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
		Teacher.shuffle(nums, 10); // Тасование последовательности
		for (int j = 0; j < 10; j++) { // Цикл прохода по обучающей выборке
			for (int u = 0; u < 3; u++) { // Количество проходов по одной цифре
				// Работа сети
				// Обнуление переменной максимума
				max = 0;
				// Считывание картика поданной на вход сети
				IMAGE_1 = Nums[NUMBER][i];
				// Проход картинки через первый сверточный слой
				for (int l = 0; l < f1_count; l++) {
					IMAGE_2[l] = NeyronCNN.Svertka(FILTERS[l], IMAGE_1);
				}
				// Операция макспулинга
				for (int l = 0; l < f1_count; l++) {
					IMAGE_3[l] = NeyronCNN.Pooling(IMAGE_2[l], 2, 2);
				}
				// Проход картинки через второй сверточный слой
				for (int l = 0; l < f1_count; l++) {
					for (int ll = 0; ll < k; ll++) {
						IMAGE_4[l*k + ll] = NeyronCNN.Svertka(FILTERS1[l*k + ll], IMAGE_3[l]);
					}
				}
				// Операция макспулинга
				for (int l = 0; l < f2_count; l++) {
					IMAGE_5[l] = NeyronCNN.Pooling(IMAGE_4[l], 2, 2);
				}
				for (int l = 0; l < f2_count; l++) {
					for (int li = 0; li < 4; li++) {
						for (int lj = 0; lj < 4; lj++) {
							IMAGE_OUT[li][l * 4 + lj] = IMAGE_5[l][li][lj];
						}
					}
				}
				// Проход по перцептрону
				// Проход по первому слою
				for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
					summ = Neyron.Summator(IMAGE_OUT, WEIGHTS[0][l]); // Получение взвешенной суммы
					MATRIX_OUT[0][l] = Neyron.FunkActiv(summ, F_2);
				}
				for (int l = 0; l < w2_count; l++) { // Цикл прохода по сети
					summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]); // Получение взвешенной суммы
					y[l] = Neyron.FunkActiv(summ, F_1); // Запись выхода l-того нейрона в массив выходов сети
				}
				for (int l = 1; l < w2_count; l++) { // Получение результатов сети
					if (y[l] > y[max]) {
						max = l;
					}
				}
				// Вывод распознанной цифры на экран для визуализации процесса обучения
				/*cout << max << ' ';*/
				// Расчет ошибки
				for (int i = 0; i < w2_count; i++) {
					if (i == NUMBER)
						a[i] = 1;
					if (i != NUMBER)
						a[i] = 0;
				}
				// Вывод ошибки на экран
				/*cout << Teacher.RMS_error(a, y, w2_count) << endl;*/
				// Если ошибка мала, пропускаем цикл обучения, что бы избежать переобучения сети
				if (Teacher.RMS_error(a, y, w2_count) < 0.3) {
					continue;
				}
				// Обучение сети
				for (int l = 0; l < w2_count; l++) { // Расчет ошибки для выходного слоя
					if (l == NUMBER) { // Если номер нейрона совпадает с поданной на вход цифрой, то ожидаеммый ответ 1
						WEIGHTS1[0][l].GetD() = Teacher.PartDOutLay(1, y[l]); // Расчет ошибки
					}
					else {// Если номер нейрона совпадает с поданной на вход цифрой, то ожидаеммый ответ 1
						WEIGHTS1[0][l].GetD() = Teacher.PartDOutLay(0, y[l]); // Расчет ошибки
					}
				}
				// Распространение ошибки на скрытые слои перцептрона
				for (int l = 0; l < w2_count; l++) {
					Teacher.BackPropagation(WEIGHTS, WEIGHTS1[0][l]);
				}
				// Распространение ошибки на выход картинки
				for (int l = 0; l < w2_count; l++) {
					TeacherCNN.Revers_Perceptron_to_CNN(IMAGE_OUT_D, WEIGHTS[0][l]);
				}
				// Копирование ошибки на подвыборочный слой
				for (int l = 0; l < f2_count; l++) {
					for (int li = 0; li < 4; li++) {
						for (int lj = 0; lj < 4; lj++) {
							IMAGE_5_D[l][li][lj] = IMAGE_OUT_D[li][l * 4 + lj];
						}
					}
				}
				// Распространение ошибки на сверточный слой
				for (int l = 0; l < f2_count; l++) {
					IMAGE_4_D[l] = TeacherCNN.ReversPooling(IMAGE_5_D[l], 2, 2);
				}
				// Распространение ошибки на подвыборочный слой
				for (int l = 0; l < f1_count; l++) {
					IMAGE_3_D[l] = TeacherCNN.ReversConvolution(IMAGE_4_D[l*k], FILTERS1[l*k]);
					for (int ll = 1; ll < k; ll++) {
						IMAGE_3_D[l] = IMAGE_3_D[l] + TeacherCNN.ReversConvolution(IMAGE_4_D[l*k + ll], FILTERS1[l*k + ll]);
					}
				}
				// Распространение ошибки на сверточный слой
				for (int l = 0; l < f1_count; l++) {
					IMAGE_2_D[l] = TeacherCNN.ReversPooling(IMAGE_3_D[l], 2, 2);
				}
				// Примемение градиентного спуска 
				// Первый сверточный слой
				for (int l = 0; l < f1_count; l++) {
					TeacherCNN.GradDes(IMAGE_1, IMAGE_2_D[l], FILTERS[l]);
				}
				// Второй сверточный слой
				for (int l = 0; l < f1_count; l++) {
					for (int ll = 0; ll < k; ll++) {
						TeacherCNN.GradDes(IMAGE_3[l], IMAGE_4_D[l*k + ll], FILTERS1[l*k + ll]);
					}
				}
				// Перцептрон
				// Первый слой
				for (int l = 0; l < w1_count; l++) { // Примемение градиентного спуска по всем нейроннам первого слоя
					Teacher.GradDes(WEIGHTS[0][l], IMAGE_OUT, f_2, MATRIX_OUT[0][l]);
				}
				// Второй слой
			for (int l = 0; l < w2_count; l++) { // Примемение градиентного спуска по всем нейроннам второго слоя
					summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]);
					Teacher.GradDes(WEIGHTS1[0][l], MATRIX_OUT, f_1, summ);
				}
				// Обнуление ошибок
				for (int l = 0; l < w1_count; l++) { // Обнуление ошибки нейронов 1 слоя
					WEIGHTS[0][l].GetD() = 0;
				}
				// Обнуления вектора ошибок
				IMAGE_OUT_D.Fill(0);
				// "Замедление обучения сети"
				Teacher.getE() -= Teacher.getE() * 0.0000001;
				TeacherCNN.getE() -= TeacherCNN.getE() * 0.0000001;
			}
		}
		// Прогресс бар
		if (i > (koll / 100)* procent){
			cout << '%'<<procent<< endl;
			procent++;
		}
	}

	// Сохранение весов
	ofstream fWeights;
	fWeights.open("./resources/Weights.txt");
	for (int i = 0; i < f1_count; i++) {
		fWeights << FILTERS[i];
	}
	for (int i = 0; i < f2_count; i++) {
		fWeights << FILTERS1[i];
	}
	fWeights << WEIGHTS;
	fWeights << WEIGHTS1;
	fWeights.close();

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
	 vector<vector<Matrix<double> > > TestNums(10);
	 for (int i = 0; i < 10; i++) {
		 TestNums[i] = vector<Matrix<double> >(100);
	 }
	 // Считывание тестовой выборки
	 folder = "../Image_to_txt/resources/";
	 for (int i = 0; i < 10; i++) {
		 file = to_string(i) + ".txt";
		 path = folder + file;
		 ifstream inputt(path);
		 for (int j = 0; j < 30; j++) {
			 inputt >> TestNums[i][j];
		 }
		 inputt.close();
	 }
	// Переменная ошибок сети
	int errors_network = 0;
	// Вывод на экран реультатов тестирования сети
	cout << "Test network:" << endl;
	for (int i = 0; i < 10; i++) { // Цикл прохода по тестовой выборке
		for (int j = 0; j < 3; j++) {
			int max = 0;
			// Работа сети
			// Считывание картика поданной на вход сети
			IMAGE_1 = TestNums[i][j];
			// Проход картинки через первый сверточный слой
			for (int l = 0; l < f1_count; l++) {
				IMAGE_2[l] = NeyronCNN.Svertka(FILTERS[l], IMAGE_1);
			}
			// Операция макспулинга
			for (int l = 0; l < f1_count; l++) {
				IMAGE_3[l] = NeyronCNN.Pooling(IMAGE_2[l], 2, 2);
			}
			// Проход картинки через второй сверточный слой
			for (int l = 0; l < f1_count; l++) {
				for (int ll = 0; ll < k; ll++) {
					IMAGE_4[l*k + ll] = NeyronCNN.Svertka(FILTERS1[l*k + ll], IMAGE_3[l]);
				}
			}
			// Операция макспулинга
			for (int l = 0; l < f2_count; l++) {
				IMAGE_5[l] = NeyronCNN.Pooling(IMAGE_4[l], 2, 2);
			}
			for (int l = 0; l < f2_count; l++) {
				for (int li = 0; li < 4; li++) {
					for (int lj = 0; lj < 4; lj++) {
						IMAGE_OUT[li][l * 4 + lj] = IMAGE_5[l][li][lj];
					}
				}
			}
			// Проход по перцептрону
			// Проход по первому слою
			for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
				summ = Neyron.Summator(IMAGE_OUT, WEIGHTS[0][l]); // Получение взвешенной суммы
				MATRIX_OUT[0][l] = Neyron.FunkActiv(summ, F_1);
			}
			for (int l = 0; l < w2_count; l++) { // Цикл прохода по сети
				summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]); // Получение взвешенной суммы
				y[l] = Neyron.FunkActiv(summ, F_2); // Запись выхода l-того нейрона в массив выходов сети
			}
			for (int l = 1; l < w2_count; l++) { // Получение результатов сети
				if (y[l] > y[max]) {
					max = l;
				}
			}
			// Вывод результатов на экран
			cout << "Test " << i << " : " << "recognized " << max << ' ' << y[max] << endl;
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
			inputt >> TestNums[i][j];
		}
		inputt.close();
	}
	// Переменная количества ошибок на тестовой выборке
	int errors_resilience = 0;
	// Вывод на экран реультатов тестирования сети
	cout << "Test resilience:" << endl;
	for (int i = 0; i < 10; i++) { // Цикл прохода по тестовой выборке
		for (int j = 0; j < 100; j++) {
			max = 0;
			// Работа сети
			// Считывание картика поданной на вход сети
			IMAGE_1 = TestNums[i][j];
			// Проход картинки через первый сверточный слой
			for (int l = 0; l < f1_count; l++) {
				IMAGE_2[l] = NeyronCNN.Svertka(FILTERS[l], IMAGE_1);
			}
			// Операция макспулинга
			for (int l = 0; l < f1_count; l++) {
				IMAGE_3[l] = NeyronCNN.Pooling(IMAGE_2[l], 2, 2);
			}
			// Проход картинки через второй сверточный слой
			for (int l = 0; l < f1_count; l++) {
				for (int ll = 0; ll < k; ll++) {
					IMAGE_4[l*k + ll] = NeyronCNN.Svertka(FILTERS1[l*k + ll], IMAGE_3[l]);
				}
			}
			// Операция макспулинга
			for (int l = 0; l < f2_count; l++) {
				IMAGE_5[l] = NeyronCNN.Pooling(IMAGE_4[l], 2, 2);
			}
			for (int l = 0; l < f2_count; l++) {
				for (int li = 0; li < 4; li++) {
					for (int lj = 0; lj < 4; lj++) {
						IMAGE_OUT[li][l * 4 + lj] = IMAGE_5[l][li][lj];
					}
				}
			}
			// Проход по перцептрону
			// Проход по первому слою
			for (int l = 0; l < w1_count; l++) { // Цикл прохода по сети
				summ = Neyron.Summator(IMAGE_OUT, WEIGHTS[0][l]); // Получение взвешенной суммы
				MATRIX_OUT[0][l] = Neyron.FunkActiv(summ, F_1);
			}
			for (int l = 0; l < w2_count; l++) { // Цикл прохода по сети
				summ = Neyron.Summator(MATRIX_OUT, WEIGHTS1[0][l]); // Получение взвешенной суммы
				y[l] = Neyron.FunkActiv(summ, F_2); // Запись выхода l-того нейрона в массив выходов сети
			}
			for (int l = 1; l < w2_count; l++) { // Получение результатов сети
				if (y[l] > y[max]) {
					max = l;
				}
			}
			// Вывод результатов на экран
			cout << "Test " << i << " : " << "recognized " << max << ' ' << y[max] << endl;
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