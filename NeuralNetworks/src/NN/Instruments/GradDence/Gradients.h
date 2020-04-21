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
        explicit SGD(const double &a_=1, double y_=0.9, double p_ = DBL_MAX, double nesterov = false) :
                ImpulsGrad_speed_bordered<T>(a_, p_, "SGD"), y(y_), nesterov_(nesterov) {};

        void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history);
        void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error, size_t step, Filter<T>& history);

        ~SGD() {};
    private:
        double y;
        double nesterov_;
    };

    template<typename T>
    void SGD<T>::operator()(Neyron<T> &w, const Matrix<T>& in, Neyron<T>& history) {
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
    SGD<T>::operator()(const Tensor<T> &in, Filter<T> &F, const Matrix<T> &error, size_t step, Filter<T> &history) {
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
                ImpulsGrad_speed_bordered<T>(a_, p_, "Adagrad"){};

        void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history);
        void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error,
                size_t step, Filter<T>& history);

        ~Adagrad() {};
    };

    template<typename T>
    void Adagrad<T>::operator()(Neyron<T> &w, const Matrix<T> &in, Neyron<T> &history) {
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
                delta = (this->a * w.getError()[i][j]) / std::sqrt(history[i][j] + 0.00001);
                delta = this->clamps(delta);
                w[i][j] -= delta;
            }
        }
        g = w.getError().GetWBias() * w.getError().GetWBias();
        history.GetWBias() += g;
        delta = (this->a *  w.getError().GetWBias()) / std::sqrt(history.GetWBias() + 0.00001);
        delta = this->clamps(delta);
        w.GetWBias() -= delta;
    }

    template<typename T>
    void Adagrad<T>::operator()(const Tensor<T> &in, Filter<T> &F, const Matrix<T> &error, size_t step,
                                Filter<T> &history) {
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
                    delta = (this->a * F.getError()[k][i][j]) / std::sqrt(history[k][i][j] + 0.0000001);
                    delta = this->clamps(delta);
                    F[k][i][j] -= delta;
                }
            }
        }
    }

    template<typename T>
    class RMSProp : public ImpulsGrad_speed_bordered<T> {
    public:
        explicit RMSProp(const double &a_=1, double y_=0.9, double p_ = DBL_MAX) :
                ImpulsGrad_speed_bordered<T>(a_, p_, "RMSProp"), y(y_){};

        void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history);
        void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error,
                size_t step, Filter<T>& history);

        ~RMSProp() {};
    private:
        double y;
    };

    template<typename T>
    void RMSProp<T>::operator()(Neyron<T> &w, const Matrix<T> &in, Neyron<T> &history) {
        ImpulsGrad_speed_bordered<T>::calculateError(w,in);
        if((w.getN() != history.getN())||(w.getM() != history.getM())){
            throw std::logic_error("Матрицы нейрона и матрицы истории не совпадают!");
        }
        T delta;
        T g;
        for (int i = 0; i < w.getN(); i++) {
            for (int j = 0; j < w.getM(); j++) {
                g = w.getError()[i][j] * w.getError()[i][j];
                history[i][j] = y*history[i][j] + (1-y)*g;
                delta = (this->a * w.getError()[i][j]) / std::sqrt(history[i][j] + 0.00001);
                delta = this->clamps(delta);
                w[i][j] -= delta;
            }
        }
        g = w.getError().GetWBias() * w.getError().GetWBias();
        history.GetWBias() = y*history.GetWBias() + (1-y)*g;
        delta = (this->a *  w.getError().GetWBias()) / std::sqrt(history.GetWBias() + 0.00001);
        delta = this->clamps(delta);
        w.GetWBias() -= delta;
    }

    template<typename T>
    void RMSProp<T>::operator()(const Tensor<T> &in, Filter<T> &F, const Matrix<T> &error, size_t step,
                                Filter<T> &history) {
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
                    history[k][i][j] = y*history[k][i][j] + (1-y)*g;
                    delta = (this->a * F.getError()[k][i][j]) / std::sqrt(history[k][i][j] + 0.0000001);
                    delta = this->clamps(delta);
                    F[k][i][j] -= delta;
                }
            }
        }
    }

    template<typename T>
    class Adam : public ImpulsGrad_speed_bordered<T> {
    public:
        explicit Adam(const double &a_=0.001, double b1=0.9, double b2=0.999, double p_ = DBL_MAX) :
                ImpulsGrad_speed_bordered<T>(a_, p_, "Adam"), b1_(b1), b2_(b2), t(1) {};

        void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history);
        void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error, size_t step, Filter<T>& history);
        void endOfExample(){ t++;};
        ~Adam() {};
    private:
        double b1_, b2_;
        unsigned int t;
    };

    template<typename T>
    void Adam<T>::operator()(Neyron<T> &w, const Matrix<T> &in, Neyron<T> &history) {
        ImpulsGrad_speed_bordered<T>::calculateError(w,in);
        if((w.getN() != history.getN())||(w.getM() != history.getM())){
            throw std::logic_error("Матрицы нейрона и матрицы истории не совпадают!");
        }
        T delta;
        T g;
        T fm, sm;
        for (int i = 0; i < w.getN(); i++) {
            for (int j = 0; j < w.getM(); j++) {
                history[i][j] = b1_*history[i][j] + (1-b1_)*w.getError()[i][j];
                fm = history[i][j] / (1 - std::pow(b1_,t));
                history.getError()[i][j] = b2_ * history.getError()[i][j] +
                        (1 - b2_) * w.getError()[i][j] * w.getError()[i][j];
                sm = history.getError()[i][j] / (1 - std::pow(b2_,t));

                delta = (this->a * fm) / std::sqrt(sm + 0.000001);
                delta = this->clamps(delta);
                w[i][j] -= delta;
            }
        }
        history.GetWBias() = b1_*history.GetWBias() + (1-b1_)*w.getError().GetWBias();
        fm = history.GetWBias() / (1 - std::pow(b1_, t));
        history.getError().GetWBias() = b2_ * history.getError().GetWBias() +
                                   (1 - b2_) * w.getError().GetWBias() * w.getError().GetWBias();
        sm = history.getError().GetWBias() / (1 - std::pow(b2_,t));
        delta = (this->a * fm) / std::sqrt(sm+ 0.000001);
        delta = this->clamps(delta);
        w.GetWBias() -= delta;
    }

    template<typename T>
    void Adam<T>::operator()(const Tensor<T> &in, Filter<T> &F, const Matrix<T> &error,
            size_t step, Filter<T> &history) {
        ImpulsGrad_speed_bordered<T>::calculateError(in,error,F,step);
        T delta;
        T g;
        if((F.getHeight() != history.getHeight())||(F.getWidth() != history.getWidth())||(F.getDepth() != history.getDepth())){
            throw std::logic_error("Матрицы фильтра и матрицы истории не совпадают!");
        }
        for(int k = 0; k < F.getError().getDepth(); k++) {
            for (int i = 0; i < F.getError()[k].getN(); i++) {
                for (int j = 0; j < F.getError()[k].getM(); j++) {
                    history[k][i][j] = b1_*history[k][i][j] + (1-b1_)*F.getError()[k][i][j];

                    auto fm =  history[k][i][j] / (1-std::pow(b1_,t));
                    history.getError()[k][i][j] = b2_ * history.getError()[k][i][j] +
                                               (1 - b2_) * F.getError()[k][i][j] * F.getError()[k][i][j];
                     auto sm =  history.getError()[k][i][j] / (1-std::pow(b2_,t));
                    delta = (this->a * fm) / std::sqrt(sm + 0.000001);

                    delta = this->clamps(delta);
                    F[k][i][j] -= delta;
                }
            }
        }
    }
}

#endif //ARTIFICIALNN_GRADIENTS_H
