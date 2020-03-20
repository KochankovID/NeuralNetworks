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

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    Matrix<T> BackPropogation(const Matrix<T>& in, const Matrix<T>& out, const Matrix<T> &D, const int &n_, const int &m_);
    
    // Класс исключения ------------------------------------------------------
    class LearnFilterExeption : public std::runtime_error {
    public:
        LearnFilterExeption(std::string str) : std::runtime_error(str) {};

        ~LearnFilterExeption() {};
    };

    template <typename T>
    Matrix<T> BackPropagation(const Matrix<T> &D, const Filter<T>& in, size_t step){
        Matrix<T> new_D;
        if(step > 1) {
            new_D = D.zoom(step - 1);
        }else{
            new_D = D;
        }
        new_D = Filter<T>::Padding(new_D, in.getM()-1);
        Filter<T> F = in.roate_180();
        return F.Svertka(new_D, 1);
    }

    template<typename T>
    void GradDes(Grad<T>& G, const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step) {
        G(X, D, F, step);
    }

    template<typename T>
    inline Matrix<T> BackPropogation(const Matrix<T>& in, const Matrix<T>& out, const Matrix<T> &D, const int &n_, const int &m_) {
        if ((n_ < 0) || (m_ < 0) || (n_ > D.getN()) || (m_ > D.getM())) {
            throw LearnFilterExeption("Неверный размер ядра!");
        }

        Matrix<T> copy(D.getN() * n_, D.getM() * m_);
        bool flag;
        for (int i = 0; i < D.getN(); i++) {
            for (int j = 0; j < D.getM(); j++) {
                flag = true;
                for (int ii = i * n_; ii < i * n_ + n_; ii++) {
                    for (int jj = j * m_; jj < j * m_ + m_; jj++) {
                        if((in[ii][jj] == out[i][j])&&(flag)){
                            flag = false;
                            copy[ii][jj] = D[i][j];
                        }else{
                            copy[ii][jj] = 0;
                        }
                    }
                }
            }
        }
        return copy;
    }
}

#endif //ARTIFICIALNN_LEARNNEYRON_H
