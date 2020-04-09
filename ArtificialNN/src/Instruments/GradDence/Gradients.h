#ifndef ARTIFICIALNN_GRADIENTS_H
#define ARTIFICIALNN_GRADIENTS_H

#include "Grad.h"
#include "LearnFilter.h"
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <limits.h>
#include <math.h>

namespace ANN {

    template<typename T>
    class SGD : public ImpulsGrad_speed_bordered<T> {
    public:
        explicit SGD(const double &a_, double y_=0, double p = DBL_MAX) : ImpulsGrad_speed_bordered<T>(a_, y_) {};

        void operator()(Neyron<T> &w, const Matrix<T>& in, Neyron<T>& history) {
            calculateError(w,in);

            T delta;
            for (int i = 0; i < w.getN(); i++) {
                for (int j = 0; j < w.getM(); j++) {
                    delta = (1 - y_)*w.getError()[i][j] + y_*history[i][j];
                    delta = clamps(delta);
                    w[i][j] -= delta;
                    history[i][j] = delta;
                }
            }
            delta = (1-y_) * (w.GetD() * this->a) + y_ * history.GetWBias();
            delta = clamps(delta);
            w.GetWBias() -= delta;
            history.GetWBias() = delta;
        }

        virtual void operator()(Filter<T> &F, const Matrix<T> &error, size_t step, Tensor<T>& history) {
            T delta;
            for(int k = 0; k < F.getD().getDepth(); k++) {
                for (int i = 0; i < F.getD()[k].getN(); i++) {
                    for (int j = 0; j < F.getD()[k].getM(); j++) {

                        delta = (1 - y_) * this->a * F.getError()[k][i][j] + y_ * history[k][i][j];
                        delta = clamps(delta);
                        F[k][i][j] -= delta;
                        history[k][i][j] = delta;
                    }
                }
            }
        }

        ~SGD() {};
    private:
        double y_;
        double p_;

        void calculateError(const Tensor<T> &X, const Matrix<T> &error, Filter<T> &F, size_t step);
        void calculateError(Neyron<T>& neyron, const Matrix<T>& in);
    };

    template<typename T>
    void SGD<T>::calculateError(const Tensor<T> &X, const Matrix<T> &error, Filter<T> &F, size_t step) {
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
    void SGD<T>::calculateError(Neyron<T> &neyron, const Matrix<T> &in) {
        Weights<T> temp(neyron.getN(), neyron.getM());
        for (int i = 0; i < neyron.getN(); i++) {
            for (int j = 0; j < neyron.getM(); j++) {
                temp[i][j] = neyron.GetD() * this->a * in[i][j];
            }
        }
        neyron.setError(temp);
    }
}

#endif //ARTIFICIALNN_GRADIENTS_H
