#ifndef ARTIFICIALNN_LEARNNEYRON_H
#define ARTIFICIALNN_LEARNNEYRON_H

#include "Neyron.h"
#include "Functors.h"
#include <vector>

namespace ANN {
    // Метод обратного распространения ошибки
    template<typename T>
    static void BackPropagation(Matrix<Weights<T>> &w, const Weights<T> &y);

    // Метод обратного распространения ошибки
    template<typename T>
    static void BackPropagation(Matrix <Weights<T>> &w, const Matrix <Weights<T>> &y);

    // Метод градиентного спуска
    template<typename T>
    void GradDes(Func<T>& G, Weights <T> &w, Matrix <T> &in, Func<T> &F, const T &x);

    // Функция потерь
    template<typename T>
    T loss_function();

    // Функция метрики
    template<typename T>
    T metric_function(Func<T>& F, std::vector<T> out, std::vector<T> correct);

    // Метод стягивания весов
    template<typename T>
    void retract(Matrix<Weights<T>> &weights, const int &decs);

    // Метод стягивания весов
    template<typename T>
    void retract(Weights<T> &weights, const int &decs);

    // Тасование последовательности
    template<typename T>
    void shuffle(int *arr, const int &lenth);


    // Класс исключения ------------------------------------------------------
    class LearningExeption : public std::runtime_error {
    public:
        LearningExeption(std::string str) : std::runtime_error(str) {};

        ~LearningExeption() {};
    };

    template<typename T>
    void BackPropagation(Matrix <Weights<T>> &w, const Weights <T> &y) {
        for (int i = 0; i < y.getN(); i++) {
            for (int j = 0; j < y.getM(); j++) {
                w[i][j].GetD() += (y[i][j] * y.GetD());
            }
        }
    }

    template<typename T >
    void BackPropagation(Matrix <Weights<T>> &w, const Matrix <Weights<T>> &y) {
        for (int o = 0; o < y.getN(); o++) {
            for (int u = 0; u < y.getM(); u++) {
                for (int i = 0; i < y[o][u].getN(); i++) {
                    for (int j = 0; j < y[o][u].getM(); j++) {
                        w[i][j].GetD() += (y[o][u][i][j] * y.GetD());
                    }
                }
            }
        }
    }

    template<typename T >
    void GradDes(Func<T>& G, Weights <T> &w, Matrix <T> &in, Func<T> &F, const T &x) {
        if ((w.getN() != in.getN()) || (w.getM() != in.getM())) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        G(Weights <T> &w, Matrix <T> &in, Func<T> &F, const T &x);
    }
}

#endif //ARTIFICIALNN_LEARNNEYRON_H
