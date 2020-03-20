#ifndef ARTIFICIALNN_LEARNNEYRON_H
#define ARTIFICIALNN_LEARNNEYRON_H

#include "Filter.h"
#include "Gradients.h"

namespace ANN {
    // Метод обратного распространения ошибки
    template <typename T>
    Matrix<T> BackPropagation(const Matrix<T> &D, const Filter<T>& in, size_t step);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(Grad<T>& G, const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step);

    // Класс исключения ------------------------------------------------------
    class LearnFilterExeption : public std::runtime_error {
    public:
        LearnFilterExeption(std::string str) : std::runtime_error(str) {};

        ~LearnFilterExeption() {};
    };

    template <typename T>
    Matrix<T> BackPropagation(const Matrix<T> &D, const Filter<T>& in, size_t step){
        Matrix<T> new_D;
        if(step > 1){
            new_D = D.zoom(step-1);
        }else{
            new_D = Filter<T>::Padding(D, in.getM()-1);
        }
        Filter<T> F = in.roate_180();
        return F.Svertka(new_D, 1);
    }

    template<typename T>
    void GradDes(Grad<T>& G, const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step) {
        G(X, D, F, step);
    }
}

#endif //ARTIFICIALNN_LEARNNEYRON_H
