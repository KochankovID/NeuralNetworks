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
        void calculateError(const Tensor<T> &X, const Matrix<T> &error, Filter<T> &F, size_t step) const;
        void calculateError(Neyron<T>& neyron, const Matrix<T>& in) const;
    };

    template<typename T>
    void SGD<T>::calculateError(const Tensor<T> &X, const Matrix<T> &error, Filter<T> &F, size_t step) const {

        Matrix<T> new_D = PrepForStepM(error, step);
        Tensor<T> temp(F.getHeight(), F.getWidth(), F.getDepth());

        for(size_t i = 0; i < F.getDepth(); i++){

            auto delta = Filter<T>::Svertka(X[i],new_D,1);
            if((delta.getN() != F.getHeight())||(delta.getM() != F.getWidth())){
                throw std::logic_error("Матрицы фильтра и матрицы ошибки не совпадают!");
            }
            temp[i] = delta;
        }

        F.setError(temp);
    }

    template<typename T>
    void SGD<T>::calculateError(Neyron<T> &neyron, const Matrix<T> &in) const {
        if((in.getN() != neyron.getN())||(in.getM() != neyron.getM())){
            throw std::runtime_error("Size of input matrix and neyron matrix is not equal!");
        }
        Weights<T> temp(neyron.getN(), neyron.getM());
        for (int i = 0; i < neyron.getN(); i++) {
            for (int j = 0; j < neyron.getM(); j++) {
                temp[i][j] = neyron.GetD() * this->a * in[i][j];
            }
        }
        temp.GetWBias() = neyron.GetD() * this->a;
        neyron.setError(temp);
    }

    template<typename T>
    void SGD<T>::operator()(Neyron<T> &w, const Matrix<T>& in, Neyron<T>& history) const {
        calculateError(w,in);
        if((w.getN() != history.getN())||(w.getM() != history.getM())){
            throw std::logic_error("Матрицы нейрона и матрицы истории не совпадают!");
        }
        T delta;
        for (int i = 0; i < w.getN(); i++) {
            for (int j = 0; j < w.getM(); j++) {
                delta = (1 - y)*w.getError()[i][j] + y*history[i][j];
                delta = this->clamps(delta);
                w[i][j] -= delta;
                history[i][j] = delta;
                if(nesterov_){
                    w[i][j] -= y*history[i][j];
                }
            }
        }
        delta = (1-y) * w.getError().GetWBias() + y * history.GetWBias();
        delta = this->clamps(delta);
        w.GetWBias() -= delta;
        history.GetWBias() = delta;
    }

    template<typename T>
    void
    SGD<T>::operator()(const Tensor<T> &in, Filter<T> &F, const Matrix<T> &error, size_t step, Filter<T> &history) const {
        calculateError(in,error,F,step);
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
                        F[k][i][j] -= y * history[k][i][j];
                    }
                }
            }
        }
    }
}

#endif //ARTIFICIALNN_GRADIENTS_H
