//: Нейросеть распознающая все цифры

#include "Model.h"
#include <vector>
#include <iostream>
#include <fstream>


using namespace std;
using namespace NN;
// Макрос режима работы программы (с обучением или без)

#define Teach

// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;
using namespace NN;
int main()
{
    // Создание функции ошибки
    RMS_errorD<double> rmsErrorD;

    // Создание метрики
    auto b = Accuracy<double>();
    auto c = RMS_error<double>();

    vector<Metr<double>*> metrixes;
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

	D_DenceLayer dence1(128,300,F_2, f_2,I3);
	D_DenceLayer dence2(84, 128,F_2,f_2,I4);
	D_DenceLayer dence3(10, 84,F_1,f_1,I5);

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
	Matrix<Tensor<double>> train_data;
	Matrix<Tensor<double>> train_out;

	// Считывание обучающей выборки
    auto data_set = NN::getImageDataFromDirectory<double>("./../../../mnist_png/training/",
                                                           cv::IMREAD_GRAYSCALE, 1.0/255);
    train_data = data_set.first;
    train_out = data_set.second;

    Classifier.getWeight();
	Classifier.learnModel(train_data, train_out, 10, 1, G, rmsErrorD, metrixes);
	Classifier.saveWeight();

#else
    Classifier.getWeight();
#endif // Teach

	 // Создание тестовой выборки
	 Matrix<Tensor<double> > test_data;
	 Matrix<Tensor<double> > test_out;

	 // Считывание тестовой выборки
    auto test_data_set = NN::getImageDataFromDirectory<double>("./../../../mnist_png/testing/",
                                                       cv::IMREAD_GRAYSCALE, 1.0/255);

    test_data = test_data_set.first;
    test_out = test_data_set.second;

	// Переменная ошибок сети
	int errors_network = 0;
//	// Вывод на экран реультатов тестирования сети
    auto result = Classifier.predict(test_data);

    for(size_t i = 0; i < result.getM(); i++){
        if(getIndexOfMaxElem(result[0][i][0][0], result[0][i][0][0]+10) !=
                getIndexOfMaxElem(test_out[0][i][0][0], test_out[0][i][0][0]+10)){
            errors_network ++;
        }
    }
	// Вывод количества ошибок на экран
	cout << errors_network << endl;

	return 0;

}