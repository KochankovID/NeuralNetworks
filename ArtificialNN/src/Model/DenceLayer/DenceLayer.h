#ifndef ARTIFICIALNN_DENCELAYER_H
#define ARTIFICIALNN_DENCELAYER_H

#include "Neyrons.h"
#include "LearnNeyron.h"
#include "Initializers.h"
#include "Data.h"
#include <string>

namespace ANN {

    template<typename T>
    class DenceLayer : public Matrix<Neyron<T> >{
    public:
        DenceLayer(size_t number_neyrons, size_t number_input, const Func<T>& F, const Func<T>& FD, const Init<T>& I);
        DenceLayer(const DenceLayer& copy);

        Matrix<T> passThrough(const Matrix<T>& in);

        void getWeightsFromFile(const std::string& file_name);
        void saveWeightsToFile(const std::string& file_name);

        void BackPropagation(const DenceLayer<T>& y);
        void BackPropagation(const Matrix<T>& y);

        void setZero();

        void SimpleLearning(const Matrix<T>& a, const Matrix<T>& y, const Matrix<T>& in, double speed);

        void GradDes(Grad<T>& G, const Matrix <T>& in);

        ~DenceLayer()= default;
        
    private:
        const Func<T>* F_;
        const Func<T>* FD_;
        const Init<T>* I_;
    };

    template <typename T>
    DenceLayer<T>::DenceLayer(size_t number_neyrons, size_t number_input, const Func<T>& F, const Func<T>& FD,
            const Init<T>& I) : Matrix<Neyron<T> >(1, number_neyrons){
        this->F_ = &F;
        this->FD_= &FD;
        this->I_ = &I;

        for (size_t i = 0; i < number_neyrons; i++) {
            this->arr[0][i] = Neyron<T>(1, number_input);
            for (size_t j = 0; j < number_input; j++) {
                this->arr[0][i][0][j] = (*(this->I_))();
            }
            this->arr[0][i].GetWBias() = (*(this->I_))();
        }
    }

    template <typename T>
    DenceLayer<T>::DenceLayer(const DenceLayer& copy):Matrix<Neyron<T> >(copy){
        this->F_ = copy.F_;
        this->FD_ = copy.FD_;
        this->I_ = copy.I_;
    }

    template <typename T>
    Matrix<T> DenceLayer<T>::passThrough(const Matrix<T>& in){
        Matrix<T> out(1,this->m);
        for (size_t i = 0; i < this->m; i++) { // Цикл прохода по сети
            out[0][i] = Neyron<T>::FunkActiv(this->arr[0][i].Summator(in), *(this->F_));
        }
        return out;
    }

    template <typename T>
    void DenceLayer<T>::getWeightsFromFile(const std::string& file_name){
        ANN::getWeightsTextFile(*this, file_name);
    }

    template<typename T>
    void DenceLayer<T>::saveWeightsToFile(const std::string &file_name) {
        ANN::saveWeightsTextFile(*this, file_name);
    }

    template<typename T>
    void DenceLayer<T>::BackPropagation(const DenceLayer<T> &y) {
        ANN::BackPropagation(*this, y);
    }

    template<typename T>
    void DenceLayer<T>::BackPropagation(const Matrix<T> &y) {
        ANN::BackPropagation(*this, y);
    }

    template<typename T>
    void DenceLayer<T>::GradDes(Grad<T> &G, const Matrix<T> &in) {
        ANN::GradDes(G, *this, in, *this->FD_);
    }

    template<typename T>
    void DenceLayer<T>::SimpleLearning(const Matrix<T> &a, const Matrix<T> &y, const Matrix<T> &in, double speed) {
        if((a.getN() != y.getN())||(a.getM() != y.getM())){
            throw LearningExeption("Несовпадение размеров матрицы выхода и матрицы ожидаемых выходов!");
        }
        for(size_t i = 0; i < a.getN(); i++){
            for(size_t j = 0; j < a.getM(); j++){
                ANN::SimpleLearning(a[i][j], y[i][j],this->arr[i][j], in, speed);
            }
        }
    }

    template<typename T>
    void DenceLayer<T>::setZero() {
        for(size_t i = 0; i < this->n; i++){
            for(size_t j = 0; j < this->m; j++){
                this->arr[i][j].GetD() = 0;
            }
        }
    }

}

#endif //ARTIFICIALNN_DENCELAYER_H
