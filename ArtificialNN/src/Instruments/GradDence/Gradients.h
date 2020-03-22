#ifndef ARTIFICIALNN_GRADIENTS_H
#define ARTIFICIALNN_GRADIENTS_H

#include "Grad.h"
#include <algorithm>
#include "opencv2/opencv.hpp"

#include <math.h>

namespace ANN {

    template<typename T>
    class SimpleGrad : public Grad_speed<T> {
    public:
        explicit SimpleGrad(const double &a_) : Grad_speed<T>(a_) {};

        void operator()(Neyron<T> &w, const Matrix<T> &in, const Func<T> &F) {
            T x = w.Summator(in);
            cv::parallel_for_(cv::Range(0, w.getN()), [&](const cv::Range &range) {
                T delta;
                for (int i = range.start; i < range.end; i++) {
                    for (int j = 0; j < w.getM(); j++) {
                        delta = (w.GetD() * this->a * F(x) * in[i][j]);
                        if(delta > 10){
                            delta = 10;
                        }
                        if(delta < -10){
                            delta = -10;
                        }
                        w[i][j] -= delta;
                    }
                }
            });
            T delta;
            delta = this->a * F(x) * w.GetD();
            if(delta > 10){
                delta = 10;
            }
            if(delta < -10){
                delta = -10;
            }
            w.GetWBias() -= delta;
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
                    if(delta > 10){
                        delta = 10;
                    }
                    if(delta < -10){
                        delta = -10;
                    }
                    F[i][j] -= delta;
                }
            }
        }

        ~SimpleGrad() {};
    };
}

#endif //ARTIFICIALNN_GRADIENTS_H
