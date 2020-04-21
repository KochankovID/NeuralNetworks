//: Сверточная нейросеть распознающая все цифры

#include "Model.h"
#include <vector>
#include <iostream>
#include <fstream>


using namespace std;
using namespace NN;

// Макрос режима работы программы (с обучением или без)
#define Teach

using namespace std;
using namespace NN;

int main()
{
    // Создание сверточной нейросети
    D_Model Classifier;
    Classifier.add(make_shared<D_ConvolutionLayer>(6,5,5,1,D_glorot_uniform(25,37), 1));
    Classifier.add(make_shared<D_MaxpoolingLayer>(2,2));
    Classifier.add(make_shared<D_ConvolutionLayer>(12,3,3,6,D_glorot_uniform(54,27), 1));
    Classifier.add(make_shared<D_MaxpoolingLayer>(2, 2));
    Classifier.add(make_shared<D_FlattenLayer>(5,5,12));
    Classifier.add(make_shared<D_DenceLayer>(128, 300, make_shared<D_Relu>(), make_shared<D_ReluD>(), D_glorot_uniform(300,128)));
    Classifier.add(make_shared<D_DenceLayer>(84, 128, make_shared<D_Relu>(), make_shared<D_ReluD>(), D_glorot_uniform(84,128)));
    Classifier.add(make_shared<D_DenceLayer>(10, 84, make_shared<D_Sigm>(), make_shared<D_SigmD>(), D_glorot_uniform(10,84)));

    // Создание метрики
    vector<shared_ptr<D_Metr>> metrics;
    metrics.push_back(make_shared<D_Accuracy>());
    metrics.push_back(make_shared<D_RMS_error>());

#ifdef Teach
    // Cоздание обучающей выборки
    Matrix<D_Tensor> train_data;
    Matrix<D_Tensor> train_out;

    // Считывание обучающей выборки
    auto data_set = NN::getImageDataFromDirectory<double>("./../../../mnist_png/training/",
                                                           cv::IMREAD_GRAYSCALE, 1.0/255);
    train_data = data_set.first;
    train_out = data_set.second;

    // Обучение модели
    Classifier.learnModel(train_data, train_out, 10, 1,
            make_shared<D_Adagrad>() , D_RMS_errorD(), metrics);

    // Сохранение весов модели
    Classifier.saveWeight();

#else
    // Считывание весов модели
    Classifier.getWeight();

#endif // Teach
     // Создание тестовой выборки
     Matrix<D_Tensor > test_data;
     Matrix<D_Tensor > test_out;

     // Считывание тестовой выборки
    auto test_data_set = NN::getImageDataFromDirectory<double>("./../../../mnist_png/testing/",
                                                       cv::IMREAD_GRAYSCALE, 1.0/255);

    test_data = test_data_set.first;
    test_out = test_data_set.second;

    // Оценка модели на тестовой выборке
    Classifier.evaluate(test_data, test_out, metrics);

    return 0;
}