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

    // Метод градиентного спуска
    template<typename T>
    void GradDes(Func<T>& G, Weights <T> &w, Matrix <T> &in, Func<T> &F, const T &x);

    // Функция потерь
    template<typename T>
    T loss_function(Metr<T>& F, std::vector<T> out, std::vector<T> correct);

    // Функция метрики
    template<typename T>
    T metric_function(Metr<T>& F, std::vector<T> out, std::vector<T> correct);

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

    template<typename T >
    T loss_function(Metr<T>& F, std::vector<T> out, std::vector<T> correct) {
        if (out.size() != correct.size()) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        F(std::vector<T> out, std::vector<T> correct);
    }

    template<typename T >
    T metric_function(Metr<T>& F, std::vector<T> out, std::vector<T> correct){
        if (out.size() != correct.size()) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        F(std::vector<T> out, std::vector<T> correct);
    }

    template<typename T >
    void retract(Matrix <Weights<T>> &weights, const int &decs) {
        int d = 1;
        for (int i = 0; i < decs; i++) {
            d *= 0.1;
        }
        for (int i = 0; i < weights.getN(); i++) {
            for (int j = 0; j < weights.getM(); j++) {
                for (int k = 0; k < weights[i][j].getN(); k++) {
                    for (int y = 0; y < weights[i][j].getM(); y++) {
                        if (weights[i][j][k][y] > 0) {
                            weights[i][j][k][y] -= d;
                        } else {
                            weights[i][j][k][y] += d;
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    void retract(Weights <T> &weights, const int &decs) {
        int d = 1;
        for (int i = 0; i < decs; i++) {
            d *= 0.1;
        }
        for (int k = 0; k < weights.getN(); k++) {
            for (int y = 0; y < weights.getM(); y++) {
                if (weights[k][y] > 0) {
                    weights[k][y] -= d;
                } else {
                    weights[k][y] += d;
                }
            }
        }
    }

    template<typename T, typename Y>
    void shuffle(int *arr, const int &lenth) {
        srand(time(0));
        int j = 0;
        int tmp = 0;
        for (int i = 0; i < lenth; i++) {
            j = ((double) rand() / std::numeric_limits<int>::max() * lenth);
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }


}

#endif //ARTIFICIALNN_LEARNNEYRON_H
