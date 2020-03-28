#ifndef ARTIFICIALNN_LEARNFILTER_H
#define ARTIFICIALNN_LEARNFILTER_H

#include "Filter.h"
#include "Gradients.h"

namespace ANN {
    // Метод обратного распространения ошибки
    template <typename T>
    void BackPropagation(const Matrix<T> &error, const Filter<T>& filter, size_t step, Tensor<T>& out);

    // Метод обратного распространения ошибки
    template <typename T>
    void BackPropagation(const Tensor<T> &error, const Matrix<Filter<T> >& filter, size_t step, Tensor<T>& out);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(Grad<T>& G, const Tensor<T> &input, const Matrix<T> &error, Filter<T> &filter, size_t step);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(Grad<T>& G, const Tensor<T> &input, const Tensor<T> &error,
            Matrix<Filter<T> > &filter, size_t step);

    template <typename T>
    void GradDes(ImpulsGrad<T>& G, const Tensor<T> &input, const Matrix<T> &error, Filter<T> &filter,
            size_t step, Tensor<T>& history);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(ImpulsGrad<T>& G, const Tensor<T> &input, const Tensor<T> &error,
                 Matrix<Filter<T> > &filter, size_t step, Matrix<Tensor<T>>& history);

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    void BackPropagation(const Matrix<T>& input, const Matrix<T>& output, const Matrix<T> &error,
            const int &n_, const int &m_, Matrix<T>& result);

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    void BackPropagation(const Tensor<T>& input, const Tensor<T>& output,
            const Tensor<T> &error, const int &n_, const int &m_, Tensor<T>& result);

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
        new_D = Filter<T>::Padding(new_D, filter.getWidth() - 1);
        Filter<T> F = filter.roate_180();
        for(size_t i = 0; i < result.getDepth(); i++){
            result[i] += Filter<T>::Svertka(new_D, F[i], 1);
        }
    }

    template <typename T>
    void BackPropagation(const Tensor<T> &error, const Matrix<Filter<T> >& filter, size_t step, Tensor<T>& out){
        for(size_t i = 0; i < filter.getM(); i++){
            if(filter[0][i].getDepth() != out.getDepth()){
                throw LearnFilterExeption("Глубина матрицы фильтра не равна глубине выходного тензора!");
            }
            BackPropagation(error[i], filter[0][i], step, out);
        }
    }

    template<typename T>
    void GradDes(Grad<T>& G, const Tensor<T> &input, const Matrix<T> &error, Filter<T> &filter, size_t step) {
        G(input, error, filter, step);
    }

    template <typename T>
    void GradDes(Grad<T>& G, const Tensor<T> &input, const Tensor<T> &error,
                 Matrix<Filter<T> > &filter, size_t step){

        if(filter.getM() != error.getDepth()){
            throw LearnFilterExeption("Глубина тензора выхода не совпадает с глубиной фильтров!");
        }
        for(size_t i = 0; i < filter.getM(); i++){
            G(input, error[i], filter[0][i], step);
        }
    }

    template<typename T>
    void
    ANN::GradDes(ImpulsGrad <T> &G, const Tensor <T> &input, const Matrix <T> &error, Filter <T> &filter,
                 size_t step, Tensor<T>& history) {
        G(input, error, filter, step, history);
    }

    template <typename T>
    void GradDes(ImpulsGrad<T>& G, const Tensor<T> &input, const Tensor<T>&error,
                 Matrix<Filter<T> > &filter, size_t step, Matrix<Tensor<T>>& history){

        if(filter.getM() != history.getM()){
            throw LearnFilterExeption("Кол-во историй и фильтров не совпадают!");
        }
        for(size_t i = 0; i < filter.getM(); i++){
            G(input, error[i], filter[i], step, history[0][i]);
        }
    }

    template<typename T>
    inline void BackPropagation(const Matrix<T>& input, const Matrix<T>& output, const Matrix<T> &error,
            const int &n_, const int &m_, Matrix<T>& result) {
        if ((n_ < 0) || (m_ < 0) || (n_ > error.getN()) || (m_ > error.getM())) {
            throw LearnFilterExeption("Неверный размер ядра!");
        }

        bool flag;
        for (int i = 0; i < error.getN(); i++) {
            for (int j = 0; j < error.getM(); j++) {
                flag = true;
                for (int ii = i * n_; ii < i * n_ + n_; ii++) {
                    for (int jj = j * m_; jj < j * m_ + m_; jj++) {
                        if((input[ii][jj] == output[i][j])&&(flag)){
                            flag = false;
                            result[ii][jj] = error[i][j];
                        }else{
                            result[ii][jj] = 0;
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void BackPropagation(const Tensor<T>& input, const Tensor<T>& out,
                              const Tensor<T> &error, const int &n_, const int &m_, Tensor<T>& result){

        if((input.getDepth() != out.getDepth())||(out.getDepth() != error.getDepth())||
        (input.getDepth() != out.getDepth())||(out.getDepth() != error.getDepth())){
            throw LearnFilterExeption("Несовпадение размеров матриц!");
        }

        for(size_t i = 0; i < result.getDepth(); i++){
            ANN::BackPropagation(input[i], out[i], error[i], n_, m_, result[i]);
        }
    }
}

#endif //ARTIFICIALNN_LEARNFILTER_H
