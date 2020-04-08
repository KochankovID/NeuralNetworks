#ifndef ARTIFICIALNN_GRADIENTS_H
#define ARTIFICIALNN_GRADIENTS_H

#include "Grad.h"
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <limits.h>
#include <math.h>

namespace ANN {

    template<typename T>
    class SGD : public Grad_speed<T> {
    public:
        explicit SGD(const double &a_, double p = DBL_MAX) : Grad_speed<T>(a_), p_(p) {};

        void operator()(Neyron<T> &w, const Matrix<T> &in) {
            T delta;
            for (int i = 0; i < w.getN(); i++) {
                for (int j = 0; j < w.getM(); j++) {
                    delta = w.GetD() * in[i][j] * this->a;
                    if(delta > p_){
                        delta = p_;
                    }
                    if(delta < -p_){
                        delta = -p_;
                    }
                    w[i][j] -= delta;
                }
            }
            w.GetWBias() -= w.GetD() * this->a;
        }

        virtual void operator()(Filter<T> &F) {

            if((F.getD().getHeight() != F.getHeight())||(F.getD().getWidth() != F.getWidth())||(F.getD().getDepth() != F.getDepth())){
                throw std::logic_error("Матрицы фильтра и матрицы ошибки не совпадают!");
            }

            T delta;
            for(size_t k = 0; k < F.getD().getDepth(); k++) {
                for (int i = 0; i < F.getD()[k].getN(); i++) {
                    for (int j = 0; j < F.getD()[k].getM(); j++) {
                        delta = this->a * F.getD()[k][i][j];
                        if (delta > p_) {
                            delta = p_;
                        }
                        if (delta < -p_) {
                            delta = -p_;
                        }
                        F[k][i][j] -= delta;
                    }
                }
            }
        }

        ~SGD() {};
    private:
        double p_;
    };

    template<typename T>
    class SGD_Momentum : public ImpulsGrad_speed<T> {
    public:
        explicit SGD_Momentum(const double &a_, double y, double p = DBL_MAX) : ImpulsGrad_speed<T>(a_), y_(y), p_(p) {};

        void operator()(Neyron<T> &w, const Matrix<T> &in, Neyron<T>& history) {
            T delta;
            for (int i = 0; i < w.getN(); i++) {
                for (int j = 0; j < w.getM(); j++) {
                    delta = (1 - y_)*(w.GetD() * in[i][j] * this->a) + y_*history[i][j];
                    if(delta > p_){
                        delta = p_;
                    }
                    if(delta < -p_){
                        delta = -p_;
                    }
                    w[i][j] -= delta;
                    history[i][j] = delta;
                }
            }
            delta = (1-y_) * (w.GetD() * this->a) + y_ * history.GetWBias();
            w.GetWBias() -= delta;
            history.GetWBias() = delta;
        }

        virtual void operator()(Filter<T> &F, Tensor<T>& history) {

            if((F.getD()[0].getN() != F.getHeight())||(F.getD()[0].getM() != F.getWidth())){
                throw std::logic_error("Матрицы фильтра и матрицы ошибки не совпадают!");
            }

            T delta;
            for(int k = 0; k < F.getD().getDepth(); k++) {
                for (int i = 0; i < F.getD()[k].getN(); i++) {
                    for (int j = 0; j < F.getD()[k].getM(); j++) {

                        delta = (1 - y_) * this->a * F.getD()[k][i][j] + y_ * history[k][i][j];
                        if (delta > p_) {
                            delta = p_;
                        }
                        if (delta < -p_) {
                            delta = -p_;
                        }
                        F[k][i][j] -= delta;
                        history[k][i][j] = delta;
                    }
                }
            }
        }

        ~SGD_Momentum() {};
    private:
        double y_;
        double p_;
    };
}

#endif //ARTIFICIALNN_GRADIENTS_H
