#ifndef ARTIFICIALNN_LEARNNEYRON_H
#define ARTIFICIALNN_LEARNNEYRON_H

#include "Neuron.h"
#include "Functors.h"
#include "Metrics.h"
#include "Gradients.h"
#include <vector>

namespace NN {
    // Метод обратного распространения ошибки для одного нейрона и одного значения ошибки
    template <typename T>
    void BackPropagation(Neuron<T> &neyron, T error, T derivative);

    // Метод обратного распространения ошибки для матрицы нейронов и матрицы значения ошибки
    template<typename T>
    void BackPropagation(Matrix<Neuron<T>> &neyrons, const Matrix<T> &error, const Matrix<T>& derivative);

    // Метод обратного распространения ошибки для матрицы нейронов и нейрона с уже распространненой ошибкой
    template<typename T>
    void BackPropagation(Matrix <Neuron<T>> &neyrons, const Neuron <T> &error, const Matrix<T>& derivative);

    // Метод обратного распространения ошибки для матрицы нейронов и матрицы нейронов с уже распространненой ошибкой
    template<typename T>
    void BackPropagation(Matrix <Neuron<T>> &neyrons, const Matrix <Neuron<T>> &error, const Matrix<T>& derivative);

    // Метод градиентного спуска для одного нейрона
    template<typename T>
    void GradDes(ImpulsGrad<T>& G, Neuron <T> &neyron, Matrix <T> &in, Neuron<T>& history);

    // Метод градиентного спуска для матрицы нейронов
    template<typename T>
    void GradDes(ImpulsGrad<T>& G, Matrix<Neuron<T> > &neyrons, const Matrix <T> &in, Matrix<Neuron<T>>& history,
                 double dropout_rate = 0);

    // Метод обучения по правилу перцептрона
    template<typename T>
    void SimpleLearning(const T& a, const T& y, Neuron<T>& neyron, const Matrix<T>& in, double speed);

    // Функция потерь
    template<typename T>
    Matrix<double > loss_function(const Metr<T>& F, const Matrix<T>& out, const Matrix<T>& correct);

    // Функция метрики
    template<typename T>
    double  metric_function(const Metr<T>& F, const Matrix<T>& out, const Matrix<T>& correct);

    // Метод стягивания весов
    template<typename T>
    void retract(Matrix<Neuron<T>> &Neyron, const int &decs);

    // Метод стягивания весов
    template<typename T>
    void retract(Neuron<T> &Neyron, const int &decs);

    // Класс исключения ------------------------------------------------------
    class LearningExeption : public std::runtime_error {
    public:
        LearningExeption(std::string str) : std::runtime_error(str) {};

        ~LearningExeption() {};
    };

    template<typename T>
    void BackPropagation(Neuron<T> &neyron, T error, T derivative) {
        neyron.GetD() += error * derivative;
    }

    template<typename T >
    void BackPropagation(Matrix <Neuron<T>> &neyrons, const Matrix <T> &error, const Matrix<T>& derivative) {
        if((neyrons.getN() != derivative.getN())||(neyrons.getM() != derivative.getM())){
            throw LearningExeption("Mismatch neyron's matrix and derivative's matrix!");
        }
        for (int o = 0; o < error.getN(); o++) {
            for (int u = 0; u < error.getM(); u++) {
                BackPropagation(neyrons[o][u], error[o][u], derivative[o][u]);
            }
        }
    }

    template<typename T>
    void BackPropagation(Matrix <Neuron<T>> &neyrons, const Neuron <T> &error, const Matrix<T>& derivative) {
        if((neyrons.getN() != error.getN()) || (neyrons.getM() != error.getM())){
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        for (int i = 0; i < error.getN(); i++) {
            for (int j = 0; j < error.getM(); j++) {
                neyrons[i][j].GetD() += (error[i][j] * error.GetD() * derivative[i][j]);
            }
        }
    }

    template<typename T >
    void BackPropagation(Matrix <Neuron<T>> &neyrons, const Matrix <Neuron<T>> &error, const Matrix<T>& derivative) {
        if((neyrons.getN() != derivative.getN())||(neyrons.getM() != derivative.getM())){
            throw LearningExeption("Mismatch neyron's matrix and derivative's matrix!");
        }
        for (int o = 0; o < error.getN(); o++) {
            for (int u = 0; u < error.getM(); u++) {
                BackPropagation(neyrons,error[o][u],derivative);
            }
        }
    }

    template<typename T >
    Matrix<double > loss_function(const Metr<T>& F, const Matrix<T>& out, const Matrix<T>& correct) {
        if ((out.getN() != correct.getN())||(out.getM() != correct.getM())) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        return F(out, correct);
    }

    template<typename T >
    double metric_function(const Metr<T>& F, const Matrix<T>& out, const Matrix<T>&  correct){
        if ((out.getN() != correct.getN())||(out.getM() != correct.getM())) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        return F(out, correct)[0][0];
    }

    template<typename T >
    void retract(Matrix <Neuron<T>> &Neyron, const int &decs) {
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
    void retract(Neuron <T> &Neyron, const int &decs) {
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

    template<typename T>
    void SimpleLearning(const T& a, const T& y, Neuron<T>& neyron, const Matrix<T>& in, double speed){
        if ((neyron.getN() != in.getN()) || (neyron.getM() != in.getM())) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        T delta = a - y;
        if (delta == 0) {
            return;
        }
        for (int i = 0; i < neyron.getN(); i++) {
            for (int j = 0; j < neyron.getM(); j++) {
                neyron[i][j] += delta * in[i][j] * speed;
            }
        }
        neyron.GetWBias() += delta * speed;
    }

    template<typename T>
    void GradDes(ImpulsGrad<T> &G, Neuron<T> &neyron, Matrix<T> &in, Neuron<T> &history) {
        if ((neyron.getN() != in.getN()) || (neyron.getM() != in.getM())) {
            throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
        }
        G(neyron, in, history);
    }

    template<typename T>
    void GradDes(ImpulsGrad<T> &G, Matrix<Neuron<T>> &neyrons, const Matrix<T> &in, Matrix<Neuron<T> > &history,
            double dropout_rate) {
        srand(time(0));
        for(size_t i = 0; i < neyrons.getN(); i++){
            for(size_t j = 0; j < neyrons.getM(); j++){
                if ((neyrons[i][j].getN() != in.getN()) || (neyrons[i][j].getM() != in.getM())) {
                    throw LearningExeption("Несовпадение размеров входной матрицы и матрицы весов!");
                }
                if((double(rand()) / RAND_MAX) < dropout_rate){
                    continue;
                }
                G(neyrons[i][j], in, history[i][j]);
            }
        }
    }

}

#endif //ARTIFICIALNN_LEARNNEYRON_H
