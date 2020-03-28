#ifndef ARTIFICIALNN_LEARNFILTER_H
#define ARTIFICIALNN_LEARNFILTER_H

#include "Filter.h"
#include "Gradients.h"

namespace ANN {
    // Метод обратного распространения ошибки
    template <typename T>
    void BackPropagation(const Matrix<T> &error, const Filter<T>& filter, size_t step);

    // Метод обратного распространения ошибки
    template <typename T>
    Matrix<Tensor<T>> BackPropagation(const Tensor<T> &error, const Matrix<Filter<T> >& filter, size_t step);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(Grad<T>& G, const Matrix<T> &input, const Matrix<T> &error, Filter<T> &filter, size_t step);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(Grad<T>& G, const Matrix<Matrix<T> > &input, const Matrix<Matrix<T> > &error,
            Matrix<Filter<T> > &filter, size_t step);

    template <typename T>
    void GradDes(ImpulsGrad<T>& G, const Matrix<T> &input, const Matrix<T> &error, Filter<T> &filter,
            size_t step, Matrix<T>& history);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(ImpulsGrad<T>& G, const Matrix<Matrix<T> > &input, const Matrix<Matrix<T> > &error,
                 Matrix<Filter<T> > &filter, size_t step, Matrix<Matrix<T>>& history);

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    Matrix<T> BackPropagation(const Matrix<T>& input, const Matrix<T>& output, const Matrix<T> &error,
            const int &n_, const int &m_);

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    Matrix<Matrix<T> >BackPropagation(const Matrix<Matrix<T>>& input, const Matrix<Matrix<T>>& output,
            const Matrix<Matrix<T> > &error, const int &n_, const int &m_);

    // Класс исключения ------------------------------------------------------
    class LearnFilterExeption : public std::runtime_error {
    public:
        LearnFilterExeption(std::string str) : std::runtime_error(str) {};

        ~LearnFilterExeption() {};
    };

    template <typename T>
    void BackPropagation(const Matrix<T> &error, const Filter<T>& filter, size_t step, Tensor<T>& result){
        if(result.getDepth() != filter.getDepth()){
            throw LearnFilterExeption("Mismatch result tensor and filter depth!");
        }
        Matrix<T> new_D;
        if(step > 1){
            new_D = error.zoom(step - 1);
        }else{
            new_D = error;
        }
        new_D = Filter<T>::Padding(new_D, filter.getWight()-1);
        Filter<T> F = filter.roate_180();
        for(size_t i = 0; i < result.getDepth(); i++){
            result[i] += Filter<T>::Svertka(new_D, F[i], 1);
        }
    }

    template<typename T>
    void GradDes(Grad<T>& G, const Matrix<T> &input, const Matrix<T> &error, Filter<T> &filter, size_t step) {
        G(input, error, filter, step);
    }

    template<typename T>
    inline Matrix<T> BackPropagation(const Matrix<T>& input, const Matrix<T>& output, const Matrix<T> &error,
            const int &n_, const int &m_) {
        if ((n_ < 0) || (m_ < 0) || (n_ > error.getN()) || (m_ > error.getM())) {
            throw LearnFilterExeption("Неверный размер ядра!");
        }

        Matrix<T> copy(error.getN() * n_, error.getM() * m_);
        bool flag;
        for (int i = 0; i < error.getN(); i++) {
            for (int j = 0; j < error.getM(); j++) {
                flag = true;
                for (int ii = i * n_; ii < i * n_ + n_; ii++) {
                    for (int jj = j * m_; jj < j * m_ + m_; jj++) {
                        if((input[ii][jj] == output[i][j])&&(flag)){
                            flag = false;
                            copy[ii][jj] = error[i][j];
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
    void GradDes(Grad<T>& G, const Matrix<Matrix<T> > &input, const Matrix<Matrix<T> > &error,
            Matrix<Filter<T> > &filter, size_t step){
        if((error.getN()%filter.getN())||(error.getM()%filter.getM())){
            throw LearnFilterExeption("Размер матриц не пропорционален!");
        }
        if((error.getN()/filter.getN() != input.getN())||(error.getM()%filter.getM())){
            throw LearnFilterExeption("Размер матрицы входов и матрицы выходов не пропорционален!");
        }
        for(size_t i = 0; i < error.getN(); i++){
            for(size_t j = 0; j < error.getM(); j++){
                G(input[i/filter.getN()][j/filter.getM()], error[i][j], filter[i%filter.getN()][j%filter.getM()], step);
            }
        }
    }

    template <typename T>
    Matrix<Tensor<T>> BackPropagation(const Tensor<T> &error, const Matrix<Filter<T> >& filter, size_t step){
        if((error.getN()%filter.getN())||(error.getM()%filter.getM())){
            throw LearnFilterExeption("Размер матриц не пропорционален!");
        }
        Matrix<Tensor<T>> result(1, filter.getM());

        for(size_t i = 0; i < filter.getM(); i++){
            result[0][i] = BackPropagation(error, filter[0][i], step);
        }

        return result;
    }

    template<typename T>
    void
    ANN::GradDes(ImpulsGrad <T> &G, const Matrix <T> &input, const Matrix <T> &error, Filter <T> &filter,
            size_t step, Matrix<T>& history) {
        G(input, error, filter, step, history);
    }

    template <typename T>
    void GradDes(ImpulsGrad<T>& G, const Matrix<Matrix<T> > &input, const Matrix<Matrix<T> > &error,
                 Matrix<Filter<T> > &filter, size_t step, Matrix<Matrix<T>>& history){

        if((error.getN()%filter.getN())||(error.getM()%filter.getM())){
            throw LearnFilterExeption("Размер матриц не пропорционален!");
        }
        if((error.getN()/filter.getN() != input.getN())||(error.getM()%filter.getM())){
            throw LearnFilterExeption("Размер матрицы входов и матрицы выходов не пропорционален!");
        }
        for(size_t i = 0; i < error.getN(); i++){
            for(size_t j = 0; j < error.getM(); j++){
                G(input[i/filter.getN()][j/filter.getM()], error[i][j], filter[i%filter.getN()][j%filter.getM()],
                        step, history[i%filter.getN()][j%filter.getM()]);
            }
        }
    }

    template <typename T>
    Matrix<Matrix<T> > BackPropagation(const Matrix<Matrix<T>>& input, const Matrix<Matrix<T>>& out,
                              const Matrix<Matrix<T> > &error, const int &n_, const int &m_){

        if((input.getN() != out.getN())||(out.getN() != error.getN())||(input.getM() != out.getM())||(out.getM() != error.getM())){
            throw LearnFilterExeption("Несовпадение размеров матриц!");
        }

        Matrix<Matrix<T> > result(input.getN(), input.getM());

        for(size_t i = 0; i < error.getN(); i++){
            for(size_t j = 0; j < error.getM(); j++){
                result[i][j] = BackPropagation(input[i][j], out[i][j], error[i][j], n_, m_);
            }
        }
        return result;
    }
}

#endif //ARTIFICIALNN_LEARNFILTER_H
