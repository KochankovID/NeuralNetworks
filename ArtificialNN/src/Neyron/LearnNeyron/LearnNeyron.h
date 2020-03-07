#ifndef ARTIFICIALNN_LEARNNEYRON_H
#define ARTIFICIALNN_LEARNNEYRON_H

#include "Neyron.h"
#include "Functors.h"
#include "Metrix.h"
#include "Gradients.h"
#include <vector>

namespace ANN {
    // Метод обратного распространения ошибки
    template<typename T>
    static void BackPropagation(Matrix<Neyron<T>> &w, const Neyron<T> &y);

    // Метод обратного распространения ошибки
    template<typename T>
    static void BackPropagation(Matrix <Neyron<T>> &w, const Matrix <Neyron<T>> &y);

    // Метод градиентного спуска
    template<typename T>
    void GradDes(Grad<T>& G, Neyron <T> &w, const Matrix <T> &in, Func<T> &F);

    // Функция потерь
    template<typename T>
    T loss_function(Metr<T>& F, std::vector<T> out, std::vector<T> correct);

    // Функция метрики
    template<typename T>
    T metric_function(Metr<T>& F, std::vector<T> out, std::vector<T> correct);

    // Метод стягивания весов
    template<typename T>
    void retract(Matrix<Neyron<T>> &Neyron, const int &decs);

    // Метод стягивания весов
    template<typename T>
    void retract(Neyron<T> &Neyron, const int &decs);

    // Класс исключения ------------------------------------------------------
    class LearningExeption : public std::runtime_error {
    public:
        LearningExeption(std::string str) : std::runtime_error(str) {};

        ~LearningExeption() {};
    };

    template<typename T>
    void BackPropagation(Matrix <Neyron<T>> &w, const Neyron <T> &y) {
        if((w.getN() != y.getN()) || (w.getM() != y.getM())){
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        for (int i = 0; i < y.getN(); i++) {
            for (int j = 0; j < y.getM(); j++) {
                w[i][j].GetD() += (y[i][j] * y.GetD());
            }
        }
    }

    template<typename T >
    void BackPropagation(Matrix <Neyron<T>> &w, const Matrix <Neyron<T>> &y) {
        for (int o = 0; o < y.getN(); o++) {
            for (int u = 0; u < y.getM(); u++) {
                if ((w.getN() != y[o][u].getN()) || (w.getM() != y[o][u].getM())) {
                    throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
                }
            }
        }
        for (int o = 0; o < y.getN(); o++) {
            for (int u = 0; u < y.getM(); u++) {
                for (int i = 0; i < y[o][u].getN(); i++) {
                    for (int j = 0; j < y[o][u].getM(); j++) {
                        w[i][j].GetD() += (y[o][u][i][j] * y[o][u].GetD());
                    }
                }
            }
        }
    }

    template<typename T >
    void GradDes(Grad<T>& G, Neyron <T> &w, Matrix <T> &in, Func<T> &F) {
        if ((w.getN() != in.getN()) || (w.getM() != in.getM())) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        G(w, in, F);
    }

    template<typename T >
    T loss_function(Metr<T>& F, std::vector<T> out, std::vector<T> correct) {
        if (out.size() != correct.size()) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        return F(out, correct);
    }

    template<typename T >
    T metric_function(Metr<T>& F, std::vector<T> out, std::vector<T> correct){
        if (out.size() != correct.size()) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        return F(out, correct);
    }

    template<typename T >
    void retract(Matrix <Neyron<T>> &Neyron, const int &decs) {
        double d = 0.1;
        std::pow(d, decs);
        for (int i = 0; i < Neyron.getN(); i++) {
            for (int j = 0; j < Neyron.getM(); j++) {
                for (int k = 0; k < Neyron[i][j].getN(); k++) {
                    for (int y = 0; y < Neyron[i][j].getM(); y++) {
                        if (Neyron[i][j][k][y] > 0) {
                            Neyron[i][j][k][y] -= d;
                        } else {
                            Neyron[i][j][k][y] += d;
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    void retract(Neyron <T> &Neyron, const int &decs) {
        double d = 0.1;
        std::pow(d, decs);
        for (int k = 0; k < Neyron.getN(); k++) {
            for (int y = 0; y < Neyron.getM(); y++) {
                if (Neyron[k][y] > 0) {
                    Neyron[k][y] -= d;
                } else {
                    Neyron[k][y] += d;
                }
            }
        }
    }

}

#endif //ARTIFICIALNN_LEARNNEYRON_H
