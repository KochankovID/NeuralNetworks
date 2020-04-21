#ifndef ARTIFICIALNN_LEARNFILTER_H
#define ARTIFICIALNN_LEARNFILTER_H

#include "Filter.h"
#include "Gradients.h"

namespace NN {
    template <typename T>
    Matrix<T> PrepForStepM(const Matrix<T> &D, size_t step);

    template<typename T>
    Tensor<T> PrepForStepT(const Tensor<T> &D, size_t step);

    template <typename T>
    Tensor<T> BackPropagation(const Matrix<T> &error, const Filter<T>& filter, size_t step);

    template <typename T>
    Tensor<T> BackPropagation(const Tensor<T> &error, const Matrix<Filter<T> >& filter, size_t step);

    template <typename T>
    void BackPropWeight(const Tensor<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step);

    template <typename T>
    void BackPropWeight(const Tensor<T> &X, const Tensor<T> &D, Matrix<Filter<T> > &F, size_t step);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(ImpulsGrad <T> &G, const Tensor<T>& in, Filter <T> &filter, const Matrix<T> &error, size_t step,
                 Filter<T>& history);

    // Метод градиентного спуска
    template <typename T>
    void GradDes(ImpulsGrad<T>& G, const Tensor<T>& in, Matrix<Filter<T> > &filter, const Tensor<T> &error, size_t step,
                 Matrix<Filter<T>>& history);

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    Matrix<T> BackPropagation(const Matrix<T>& input, const Matrix<T>& output, const Matrix<T> &error,
            const int &n_, const int &m_);

    // Операция обратного распространение ошибки на слое "Макс пулинга"
    template <typename T>
    Tensor<T> BackPropagation(const Tensor<T>& input, const Tensor<T>& output,
            const Tensor<T> &error, const int &n_, const int &m_);

    // Класс исключения ------------------------------------------------------
    class LearnFilterExeption : public std::runtime_error {
    public:
        LearnFilterExeption(std::string str) : std::runtime_error(str) {};

        ~LearnFilterExeption() {};
    };



    template<typename T>
    Matrix<T> PrepForStepM(const Matrix<T> &D, size_t step) {
        if(step > 1){
            return D.zoom(step - 1);
        }else{
            return D;
        }
    }

    template<typename T>
    Tensor<T> PrepForStepT(const Tensor<T> &D, size_t step) {
        if(step > 1){
            return D.zoom(step - 1);
        }else{
            return D;
        }
    }

    template<typename T>
    Tensor<T> NN::BackPropagation(const Matrix<T> &error, const Filter<T> &filter, size_t step) {
        Tensor<T> result(filter.getHeight(), filter.getWidth(), filter.getDepth());

        Matrix<T> new_D = PrepForStepM(error, step);

        new_D = Filter<T>::Padding(new_D, filter.getWidth() - 1);
        Filter<T> F = filter.roate_180();

        for(size_t i = 0; i < result.getDepth(); i++){
            result[i] = Filter<T>::Svertka(new_D, F[i], 1);
        }
        return result;
    }

    template<typename T>
    Tensor<T> NN::BackPropagation(const Tensor<T> &error, const Matrix<Filter<T>> &filter, size_t step) {
        Tensor<T> result;
        result = BackPropagation(error[0], filter[0][0], step);
        for(size_t i = 1; i < filter.getM(); i++){
            if(filter[0][i].getDepth() != result.getDepth()){
                throw LearnFilterExeption("Глубина матрицы фильтра не равна глубине выходного тензора!");
            }
            result += BackPropagation(error[i], filter[0][i], step);
        }
        return result;
    }

    template<typename T>
    void BackPropWeight(const Tensor<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step) {
        Matrix<T> new_D = PrepForStepM(D, step);

        for(size_t i = 0; i < F.getDepth(); i++){

            auto delta = Filter<T>::Svertka(X[i],new_D,1);

            if((delta.getN() != F.getHeight())||(delta.getM() != F.getWidth())){
                throw std::logic_error("Матрицы фильтра и матрицы ошибки не совпадают!");
            }

            F.getD()[i] = delta;
        }

    }

    template<typename T>
    void NN::BackPropWeight(const Tensor<T> &X, const Tensor<T> &D, Matrix<Filter<T>> &F, size_t step) {
        Tensor<T> new_D = PrepForStepT(D, step);

        for(size_t i = 0; i < F.getM(); i++){
            BackPropWeight(X,new_D[i],F[0][i], step);
        }
    }

    template<typename T>
    void GradDes(const ImpulsGrad <T> &G, const Tensor<T>& in, Filter <T> &filter, const Matrix<T> &error, size_t step,
                 Filter<T>& history) {
        G(in, filter, error, step, history);
    }

    template <typename T>
    void GradDes(ImpulsGrad<T>& G, const Tensor<T>& in, Matrix<Filter<T> > &filter, const Tensor<T> &error, size_t step,
            Matrix<Filter<T>>& history){

        if(filter.getM() != history.getM()){
            throw LearnFilterExeption("Кол-во историй и фильтров не совпадают!");
        }
        if(filter.getM() != error.getDepth()){
            throw LearnFilterExeption("Кол-во фильтров и матрикс ошибок не совпадают!");
        }
        for(size_t i = 0; i < filter.getM(); i++){
            G(in, filter[0][i], error[i], step, history[0][i]);
        }
    }

    template<typename T>
    inline Matrix<T> BackPropagation(const Matrix<T>& input, const Matrix<T>& output, const Matrix<T> &error,
            const int &n_, const int &m_) {
        if ((n_ < 0) || (m_ < 0) || (n_ > error.getN()) || (m_ > error.getM())) {
            throw LearnFilterExeption("Неверный размер ядра!");
        }
        Matrix<T> result(input.getN(), input.getM());
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
        return result;
    }

    template <typename T>
    Tensor<T> BackPropagation(const Tensor<T>& input, const Tensor<T>& out,
                              const Tensor<T> &error, const int &n_, const int &m_){

        if((input.getDepth() != out.getDepth())||(out.getDepth() != error.getDepth())){
            throw LearnFilterExeption("Несовпадение размеров матриц!");
        }

        Tensor<T> result(input.getHeight(), input.getWidth(), input.getDepth());
        for(size_t i = 0; i < result.getDepth(); i++){
           result[i] = NN::BackPropagation(input[i], out[i], error[i], n_, m_);
        }

        return result;
    }

}

#endif //ARTIFICIALNN_LEARNFILTER_H
