#ifndef ARTIFICIALNN_DENCELAYER_H
#define ARTIFICIALNN_DENCELAYER_H

#include "Neyrons.h"
#include "LearnNeyron.h"
#include "Initializers.h"
#include "Data.h"
#include "Layer.h"
#include <string>

namespace ANN {

    template<typename T>
    class DenceLayer : Layer<T>{
    public:
        DenceLayer(size_t number_neyrons, size_t number_input, const Func<T>& F, const Func<T>& FD, const Init<T>& I);
        DenceLayer(const DenceLayer& copy);

        Matrix<T> passThrough(const Matrix<T>& in);

        void getWeightsFromFile(const std::string& file_name);
        void saveWeightsToFile(const std::string& file_name);

        void BackPropagation(const DenceLayer<T>& y);
        void BackPropagation(const Matrix<T>& y);

        void Out();

        void SimpleLearning(const Matrix<T>& a, const Matrix<T>& y, const Matrix<T>& in, double speed);

        void GradDes(Grad<T>& G, const Matrix <T>& in);

        ~DenceLayer()= default;
        
    private:
        Matrix<Neyron<T>> m_;
        const Func<T>* F_;
        const Func<T>* FD_;
        const Init<T>* I_;
    };

    template <typename T>
    DenceLayer<T>::DenceLayer(size_t number_neyrons, size_t number_input, const Func<T>& F, const Func<T>& FD,
            const Init<T>& I) : Layer<T>(){
        this->m_ = Matrix<Neyron<T> >(1, number_neyrons);
        this->F_ = &F;
        this->FD_= &FD;
        this->I_ = &I;

        for (size_t i = 0; i < number_neyrons; i++) {
            this->m_[0][i] = Neyron<T>(1, number_input);
            for (size_t j = 0; j < number_input; j++) {
                this->m_[0][i][0][j] = (*(this->I_))();
            }
            this->m_[0][i].GetWBias() = (*(this->I_))();
        }
    }

    template <typename T>
    DenceLayer<T>::DenceLayer(const DenceLayer& copy){
        this->m_ = copy.m_;
        this->F_ = copy.F_;
        this->FD_ = copy.FD_;
        this->I_ = copy.I_;
    }

    template <typename T>
    Matrix<T> DenceLayer<T>::passThrough(const Matrix<T>& in){
        Matrix<T> out(1,this->m_.getM());
        for (size_t i = 0; i < this->m_.getM(); i++) { // Цикл прохода по сети
            out[0][i] = Neyron<T>::FunkActiv(this->m_[0][i].Summator(in), *(this->F_));
        }
        return out;
    }

    template <typename T>
    void DenceLayer<T>::getWeightsFromFile(const std::string& file_name){
        ANN::getWeightsTextFile(this->m_, file_name);
    }

    template<typename T>
    void DenceLayer<T>::saveWeightsToFile(const std::string &file_name) {
        ANN::saveWeightsTextFile(this->m_, file_name);
    }

    template<typename T>
    void DenceLayer<T>::BackPropagation(const DenceLayer<T> &y) {
        BackPropagation(this->m_, y.m_);
    }

    template<typename T>
    void DenceLayer<T>::BackPropagation(const Matrix<T> &y) {
        BackPropagation(this->m_, y);
    }

    template<typename T>
    void DenceLayer<T>::GradDes(Grad<T> &G, const Matrix<T> &in) {
        GradDes(G, this->m_, in, *(this->FD_));
    }

    template<typename T>
    void DenceLayer<T>::SimpleLearning(const Matrix<T> &a, const Matrix<T> &y, const Matrix<T> &in, double speed) {
        if((a.getN() != y.getN())||(a.getM() != y.getM())){
            throw LearningExeption("Несовпадение размеров матрицы выхода и матрицы ожидаемых выходов!");
        }
        for(size_t i = 0; i < a.getN(); i++){
            for(size_t j = 0; j < a.getM(); j++){
                ANN::SimpleLearning(a[i][j], y[i][j],this->m_[i][j], in, speed);
            }
        }
    }

    template<typename T>
    void DenceLayer<T>::Out() {
        std::cout << this->m_;
    }

}

#endif //ARTIFICIALNN_DENCELAYER_H
