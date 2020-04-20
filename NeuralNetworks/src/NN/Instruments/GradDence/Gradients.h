#ifndef ARTIFICIALNN_GRADIENTS_H
#define ARTIFICIALNN_GRADIENTS_H

#include "Grad.h"
#include "LearnFilter.h"
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <limits.h>
#include <math.h>

namespace NN {

    template<typename T>
    class SGD : public ImpulsGrad_speed_bordered<T> {
    public:
        explicit SGD(const double &a_=1, double y_=0, double p_ = DBL_MAX, double nesterov = false) :
        ImpulsGrad_speed_bordered<T>(a_, p_), y(y_), nesterov_(nesterov) {};

        void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history) const;
        void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error, size_t step, Filter<T>& history) const ;

        ~SGD() {};
    private:
        double y;
        double nesterov_;
    };

    template<typename T>
    void SGD<T>::operator()(Neyron<T> &w, const Matrix<T>& in, Neyron<T>& history) const {
        ImpulsGrad_speed_bordered<T>::calculateError(w,in);
        if((w.getN() != history.getN())||(w.getM() != history.getM())){
            throw std::logic_error("Матрицы нейрона и матрицы истории не совпадают!");
        }
        T delta;
        for (int i = 0; i < w.getN(); i++) {
            for (int j = 0; j < w.getM(); j++) {
                delta = (1 - y) * this->a * w.getError()[i][j] + y*history[i][j];
                delta = this->clamps(delta);
                w[i][j] -= delta;
                history[i][j] = delta;
                if(nesterov_){
                    w[i][j] -= y*history[i][j];
                }
            }
        }
        delta = (1-y) * this->a * w.getError().GetWBias() + y * history.GetWBias();
        delta = this->clamps(delta);
        w.GetWBias() -= delta;
        history.GetWBias() = delta;
    }

    template<typename T>
    void
    SGD<T>::operator()(const Tensor<T> &in, Filter<T> &F, const Matrix<T> &error, size_t step, Filter<T> &history) const {
        ImpulsGrad_speed_bordered<T>::calculateError(in,error,F,step);
        T delta;
        if((F.getHeight() != history.getHeight())||(F.getWidth() != history.getWidth())||(F.getDepth() != history.getDepth())){
            throw std::logic_error("Матрицы фильтра и матрицы истории не совпадают!");
        }
        for(int k = 0; k < F.getError().getDepth(); k++) {
            for (int i = 0; i < F.getError()[k].getN(); i++) {
                for (int j = 0; j < F.getError()[k].getM(); j++) {

                    delta = (1 - y) * this->a * F.getError()[k][i][j] + y * history[k][i][j];
                    delta = this->clamps(delta);

                    F[k][i][j] -= delta;
                    history[k][i][j] = delta;

                    if(nesterov_){
                        F[k][i][j] -= y*history[k][i][j];
                    }
                }
            }
        }
    }

    template<typename T>
    class Adagrad : public ImpulsGrad_speed_bordered<T> {
    public:
        explicit Adagrad(const double &a_=1, double p_ = DBL_MAX) :
                ImpulsGrad_speed_bordered<T>(a_, p_){};

        void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history) const;
        void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error, size_t step, Filter<T>& history) const ;

        ~Adagrad() {};
    };

    template<typename T>
    void Adagrad<T>::operator()(Neyron<T> &w, const Matrix<T> &in, Neyron<T> &history) const {
        ImpulsGrad_speed_bordered<T>::calculateError(w,in);
        if((w.getN() != history.getN())||(w.getM() != history.getM())){
            throw std::logic_error("Матрицы нейрона и матрицы истории не совпадают!");
        }
        T delta;
        T g;
        for (int i = 0; i < w.getN(); i++) {
            for (int j = 0; j < w.getM(); j++) {
                g = w.getError()[i][j] * w.getError()[i][j];
                history[i][j] += g;
                delta = (this->a * w.getError()[i][j]) / std::sqrt(history[i][j] + 0.0000001)
                delta = this->clamps(delta);
                w[i][j] -= delta;
            }
        }
        g = w.getError().GetWBias() * w.getError().GetWBias();
        history.GetWBias() += g;
        delta = (this->a *  w.getError().GetWBias()) / std::sqrt(history.GetWBias() + 0.0000001)
        delta = this->clamps(delta);
        w.GetWBias() -= delta;
    }

    template<typename T>
    void Adagrad<T>::operator()(const Tensor<T> &in, Filter<T> &F, const Matrix<T> &error, size_t step,
                                Filter<T> &history) const {
        ImpulsGrad_speed_bordered<T>::calculateError(in,error,F,step);
        T delta;
        T g;
        if((F.getHeight() != history.getHeight())||(F.getWidth() != history.getWidth())||(F.getDepth() != history.getDepth())){
            throw std::logic_error("Матрицы фильтра и матрицы истории не совпадают!");
        }
        for(int k = 0; k < F.getError().getDepth(); k++) {
            for (int i = 0; i < F.getError()[k].getN(); i++) {
                for (int j = 0; j < F.getError()[k].getM(); j++) {

                    g = F.getError()[k][i][j] * F.getError()[k][i][j];
                    history[k][i][j] += g;
                    delta = (this->a * F.getError()[k][i][j]) / std::sqrt(history[k][i][j] + 0.0000001)
                    delta = this->clamps(delta);
                    F[k][i][j] -= delta;
                }
            }
        }
    }
}

#endif //ARTIFICIALNN_GRADIENTS_H
