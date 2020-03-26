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

        virtual void operator()(const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step) {
            Matrix<T> new_D;
            if(step > 1){
                new_D = D.zoom(step-1);
            }else{
                new_D = D;
            }
            Matrix<T> error_matrix = Filter<T>::Svertka(X, new_D, 1);
            if((error_matrix.getN() != F.getN())||(error_matrix.getM() != F.getM())){
                throw std::logic_error("Матрицы фильтра и матрицы ошибки не совпадают!");
            }
            T delta;
            for (int i = 0; i < error_matrix.getN(); i++) {
                for (int j = 0; j < error_matrix.getM(); j++) {
                    delta = this->a * error_matrix[i][j];
                    if(delta > p_){
                        delta = p_;
                    }
                    if(delta < -p_){
                        delta = -p_;
                    }
                    F[i][j] -= delta;
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

        virtual void operator()(const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step,
                Matrix<T>& history) {
            Matrix<T> new_D;
            if(step > 1){
                new_D = D.zoom(step-1);
            }else{
                new_D = D;
            }
            Matrix<T> error_matrix = Filter<T>::Svertka(X, new_D, 1);
            if((error_matrix.getN() != F.getN())||(error_matrix.getM() != F.getM())){
                throw std::logic_error("Матрицы фильтра и матрицы ошибки не совпадают!");
            }
            T delta;
            for (int i = 0; i < error_matrix.getN(); i++) {
                for (int j = 0; j < error_matrix.getM(); j++) {
                    delta = (1 - y_)*this->a * error_matrix[i][j] + y_*history[i][j];
                    if(delta > p_){
                        delta = p_;
                    }
                    if(delta < -p_){
                        delta = -p_;
                    }
                    F[i][j] -= delta;
                    history[i][j] = delta;
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
