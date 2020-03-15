#ifndef ARTIFICIALNN_LEARNNEYRON_H
#define ARTIFICIALNN_LEARNNEYRON_H

#include "Filter.h"
#include "Gradients.h"

namespace ANN {
    // Метод обратного распространения ошибки
    template <typename T>
    Matrix<T> BackPropagation(const Matrix<T> &D, const Filter<T>& in, int step);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(Grad<T>& G, const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F);

    // Класс исключения ------------------------------------------------------
    class LearnFilterExeption : public std::runtime_error {
    public:
        LearnFilterExeption(std::string str) : std::runtime_error(str) {};

        ~LearnFilterExeption() {};
    };

    template <typename T>
    Matrix<T> BackPropagation(const Matrix<T> &D, const Filter<T>& in, int step){
        Matrix<T> new_D;
        if(step > 1){
            new_D = D.zoom(step-1);
        }else{
            new_D = Filter<T>::Padding(D, in.getM()-1);
        }
        Filter<T> F = in.roate_180();
        return F.Svertka(new_D, 1);
    }
}

#endif //ARTIFICIALNN_LEARNNEYRON_H
