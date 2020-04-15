//: Нейросеть распознающая все цифры

#include "Model.h"
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
    vector<Metr<double>*> metrixes;
    auto b = Accuracy<double>();
    auto c = RMS_error<double>();
    metrixes.push_back(&b);
    metrixes.push_back(&c);

    // Создание градиентного спуска
    SGD<double > G(0.09, 0.9);


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

	// Создание слоев
    D_ConvolutionLayer conv1(6, 5,5,1, I1, 1);
    D_MaxpoolingLayer maxp1(2,2);

    D_ConvolutionLayer conv2(12, 3,3,6, I2, 1);
    D_MaxpoolingLayer maxp2(2,2);

    D_FlattenLayer flat1(5,5,12);

	D_DenceLayer dence1(128,300,F_2, f_2,I3, 0.0);
	D_DenceLayer dence2(84, 128,F_2,f_2,I4, 0.0);
	D_DenceLayer dence3(10, 84,F_1,f_1,I5, 0.0);

	Model<double > Classifier;

	Classifier.add(&conv1);
	Classifier.add(&maxp1);
	Classifier.add(&conv2);
	Classifier.add(&maxp2);
	Classifier.add(&flat1);
	Classifier.add(&dence1);
	Classifier.add(&dence2);
	Classifier.add(&dence3);

#ifdef Teach
	size_t koll = 100;
	Matrix<Tensor<double>> train_data(1, koll);
	Matrix<Tensor<double>> train_out(1, koll);

	// Считывание обучающей выборки
	string folder = "../Image_to_txt/resources/";
	string path;
	string file;
	ifstream input[10];
	for (int i = 0; i < 10; i++) {
		file = to_string(i) + ".txt";
		path = folder + file;
		input[i].open(path);
	}
	int Nums[10] = {1,};
	for(int i = 0; i < koll; i++){
	    input[i%10] >> train_data[0][i];
        train_out[0][i] = D_Tensor(1,10,1);
        for(int k = 0; k < 10; k++){
            if( k == (i%10)){
                train_out[0][i][0][0][k] = 1;
            }else{
                train_out[0][i][0][0][k] = 0;
            }
        }
	}

	Classifier.learnModel(train_data, train_out, 3, 30, G, MM, metrixes);

#else
    dence1.getWeightsFromFile("./resources/Weights1.txt");
    dence2.getWeightsFromFile("./resources/Weights2.txt");
    dence3.getWeightsFromFile("./resources/Weights3.txt");
    conv1.getFiltersFromFile("./resources/Filters1.txt");
    conv2.getFiltersFromFile("./resources/Filters2.txt");

#endif // Teach

	 // Создание тестовой выборки
	 Matrix<Tensor<double> > TestNums(1, 1000);
	 Matrix<Tensor<double> > TestNums_out(1, 1000);

	 // Считывание тестовой выборки
    for (int i = 0; i < 10; i++) {
        file = to_string(i) + ".txt";
        path = folder + file;
        input[i].open(path);
    }
    for(int i = 0; i < koll; i++){
        input[i%10] >> train_data[0][i];
        TestNums_out[0][i] = D_Tensor(1,10,1);
        for(int k = 0; k < 10; k++){
            if( k == (i%10)){
                TestNums_out[0][i][0][0][k] = 1;
            }else{
                TestNums_out[0][i][0][0][k] = 0;
            }
        }
    }
	// Переменная ошибок сети
	int errors_network = 0;
//	// Вывод на экран реультатов тестирования сети
    auto result = Classifier.predict(TestNums);

    for(size_t i = 0; i < result.getM(); i++){
        if(max_element(result[0][i][0][0], result[0][i][0][0]+10) !=
        max_element(TestNums_out[0][i][0][0], TestNums_out[0][i][0][0]+10)){
            errors_network ++;
        }
    }
	// Вывод количества ошибок на экран
	cout << errors_network << endl;

//	// Считывание тестовой выборки
//	for (int i = 0; i < 10; i++) {
//		file = to_string(i) + "_tests.txt";
//		getDataFromTextFile(TestNums[0][i], "../Image_to_txt/resources/"+file);
//	}
//	// Переменная количества ошибок на тестовой выборке
//	int errors_resilience = 0;
//	// Вывод на экран реультатов тестирования сети
//	cout << "Test resilience:" << endl;
//	for (int i = 0; i < 10; i++) { // Цикл прохода по тестовой выборке
//		for (int j = 0; j < 800; j++) {
//            // Работа сети
//            // Обнуление переменной максимума
//            max = 0;
//            // Считывание картика поданной на вход сети
//            IMAGE_1[0] = TestNums[0][i][0][j];
//            // Проход картинки через первый сверточный слой
//            IMAGE_2 = conv1.passThrough(IMAGE_1);
//            // Операция макспулинга
//            IMAGE_3 = maxp1.passThrough(IMAGE_2);
//            // Проход картинки через второй сверточный слой
//            IMAGE_4 = conv2.passThrough(IMAGE_3);
//            // Операция макспулинга
//            IMAGE_5 = maxp2.passThrough(IMAGE_4);
//            // Переход со сверточных слоев к перцептрону
//            IMAGE_OUT = flat1.passThrough(IMAGE_5);
//            // Проход по перцептрону
//            // Проход по первому слою
//            MATRIX_OUT_1 = dence1.passThrough(IMAGE_OUT);
//            // Проход по второму слою
//            MATRIX_OUT_2 = dence2.passThrough(MATRIX_OUT_1);
//            MATRIX_OUT_3 = dence3.passThrough(MATRIX_OUT_2);
//            max = std::max_element(MATRIX_OUT_3[0], MATRIX_OUT_3[0]+10) - MATRIX_OUT_3[0];
//            // Вывод результатов на экран
////            cout << "Test " << i << " : " << "recognized " << max << ' ' << MATRIX_OUT_3[0][max] << endl;
//            // Подсчет ошибок
//            if (max != i) {
//				errors_resilience++;
//            }
//		}
//	}
//	// Вывод на экран реультатов тестирования сети
//	cout << errors_resilience << endl;
	return 0;

}