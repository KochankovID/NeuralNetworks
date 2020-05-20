//: Нейросеть распознающая все цифры

#include "Model.h"
#include <csv.h>
#include <iostream>

// Макрос режима работы программы (с обучением или без)
#define Teach

// Улучшение читабильности программы
#define NUMBER nums[j]

using namespace std;
using namespace NN;

int main()
{
    // Создание глубокой сети прямого распространения
    D_Model Classifier;
    Classifier.add(make_shared<D_DenceLayer>(70, 15, make_shared<D_Relu>(), make_shared<D_ReluD>(), D_glorot_uniform(15, 50)));
    Classifier.add(make_shared<D_DenceLayer>(10, 70, make_shared<D_Sigm>(), make_shared<D_SigmD>(), D_glorot_uniform(50, 10)));

    // Создание метрики
    std::vector<shared_ptr<D_Metr>> metrics;
    metrics.push_back(make_shared<D_Accuracy>());
    metrics.push_back(make_shared<D_RMS_error>());
#ifdef Teach
    // Создание обучающей выборки
    Ndarray<double> data_x({10, 15});
    Ndarray<double> data_y({10, 10});

    // Считываем матрицы обучающей выборки
    io::CSVReader<16> in("./resources/training_nums.csv");
    for (int i = 0; i < 10; i++) {
        in.read_row(data_y(i,0), data_x(i,0),
                    data_x(i,1), data_x(i,2),
                    data_x(i,3), data_x(i,4),
                    data_x(i,5), data_x(i,6),
                    data_x(i,7), data_x(i,8),
                    data_x(i,9), data_x(i,10),
                    data_x(i,11), data_x(i,12),
                    data_x(i,13), data_x(i,14));
        auto tmp = Ndarray<double>(1, 10);
        tmp.fill(0);
        tmp[int(data_y(i,0))] = 1;
        std::copy(tmp.begin(), tmp.end(), data_y.iter(1, i, 0));
    }

    // Обучение модели
    Classifier.learnModel(data_x, data_y, 1, 1000, make_shared<D_Adam>(), D_RMS_errorD(), metrics);

    // Сохранение весов сети
    Classifier.saveWeight();
#else

    // Считывание весов сети
    Classifier.getWeight();

#endif // Teach
    // Создание тестовой выборки
    Ndarray<double > test_x({90,15});
    Ndarray<double> test_y({90,10});

    // Считывание тестовой выборки из файла
    io::CSVReader<17> in_test("./resources/test_nums.csv");
    int t;
    for (int i = 0; i < 90; i++) {
        in_test.read_row(t, test_x(i,0),
                         test_x(i,1), test_x(i,2),
                         test_x(i,3), test_x(i,4),
                         test_x(i,5), test_x(i,6),
                         test_x(i,7), test_x(i,8),
                         test_x(i,9), test_x(i,10),
                         test_x(i,11), test_x(i,12),
                         test_x(i,13), test_x(i,14),
                         test_y(i,0));
        auto tmp = Ndarray<double>(1, 10);
        tmp.fill(0);
        tmp[int(test_y(i,0))] = 1;
        std::copy(tmp.begin(), tmp.end(), test_y.iter(1, i, 0));
    }

    // Оценка модели на тестовой выборке
    Classifier.evaluate(test_x, test_y, metrics);

    return 0;
}