#ifndef ARTIFICIALNN_LEARNNEYRON_H
#define ARTIFICIALNN_LEARNNEYRON_H

#include "Filter.h"
#include "Gradients.h"

namespace ANN {
    // Метод обратного распространения ошибки
    template <typename T>
    Matrix<T> BackPropagation(const Matrix<T> &D, const Filter<T>& in, size_t step);

    // Метод обратного распространения ошибки
    template <typename T>
    Matrix<Matrix<T> > BackPropagation(const Matrix<Matrix<T> > &D, const Matrix<Filter<T> >& in, size_t step);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(Grad<T>& G, const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(Grad<T>& G, const Matrix<Matrix<T> > &X, const Matrix<Matrix<T> > &D, Matrix<Filter<T> > &F, size_t step);

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    Matrix<T> BackPropogation(const Matrix<T>& in, const Matrix<T>& out, const Matrix<T> &D, const int &n_, const int &m_);

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    Matrix<Matrix<T> >BackPropogation(const Matrix<Matrix<T>>& in, const Matrix<Matrix<T>>& out,
            const Matrix<Matrix<T> > &D, const int &n_, const int &m_);

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

    template <typename T>
    void GradDes(Grad<T>& G, const Matrix<Matrix<T> > &X, const Matrix<Matrix<T> > &D, Matrix<Filter<T> > &F, size_t step){
        if((X.getN() != D.getN())||(D.getN() != F.getN())||(X.getM() != D.getM())||(D.getM() != F.getM())){
            throw LearnFilterExeption("Несовпадение размеров матриц!");
        }
        for(size_t i = 0; i < X.getN(); i++){
            for(size_t j = 0; j < X.getM(); j++){
                G(X[i][j], D[i][j], F[i][j], step);
            }
        }
    }

    template <typename T>
    Matrix<Matrix<T> > BackPropagation(const Matrix<Matrix<T> > &D, const Matrix<Filter<T> >& in, size_t step){
        if((D.getN() != in.getN())||(D.getM() != in.getM())){
            throw LearnFilterExeption("Несовпадение размеров матриц!");
        }
        Matrix<Matrix<T> > result(D.getN(), D.getM());
        for(size_t i = 0; i < D.getN(); i++){
            for(size_t j = 0; j < D.getM(); j++){
                result[i][j] = BackPropagation(D[i][j], in[i][j], step);
            }
        }
        return result;
    }

    template <typename T>
    Matrix<Matrix<T> > BackPropogation(const Matrix<Matrix<T>>& in, const Matrix<Matrix<T>>& out,
                              const Matrix<Matrix<T> > &D, const int &n_, const int &m_){

        if((in.getN() != out.getN())||(out.getN() != D.getN())||(in.getM() != out.getM())||(out.getM() != D.getM())){
            throw LearnFilterExeption("Несовпадение размеров матриц!");
        }

        Matrix<Matrix<T> > result(in.getN(), in.getM());

        for(size_t i = 0; i < D.getN(); i++){
            for(size_t j = 0; j < D.getM(); j++){
                result[i][j] = BackPropogation(in[i][j], out[i][j], D[i][j], n_, m_);
            }
        }
        return result;
    }
}

#endif //ARTIFICIALNN_LEARNNEYRON_H
