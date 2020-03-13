#ifndef ARTIFICIALNN_DENCELAYER_H
#define ARTIFICIALNN_DENCELAYER_H

#include "Neyrons.h"
#include "LearnNeyron.h"
#include <string>

namespace ANN {

    template<typename T>
    class DenceLayer {
    public:
        DenceLayer(size_t number_neyrons, Func<T> F, Func<T> FD);
        DenceLayer(const DenceLayer& copy);

        Matrix<T> passThrough(const Matrix<T>& in);

        void getWeightsFromFile(const std::string& file_name);
        void saveWeightsToFile(const std::string& file_name);

        void BackPropagation(const DenceLayer<T>& y);
        void BackPropagation(const Matrix<T>& y);

        void GradDes(Grad<T>& G, const Matrix <T>& in);


        ~DenceLayer();
    private:
        Matrix<Neyron<T>> m_;
        Func<T> F_;
        Func<T> FD_;
    };

    template <typename T>
    DenceLayer<T>::DenceLayer(size_t number_neyrons, Func<T> F, Func<T> FD){
        m_ = Matrix<Neyron<T> >(1, number_neyrons);
        F_ = F;
        FD = FD;
    }

    template <typename T>
    DenceLayer<T>::DenceLayer(const DenceLayer& copy){
        m_ = copy.m_;
        F_ = copy.F_;
        FD_ = copy.FD_;
    }

    template <typename T>
    Matrix<T> DenceLayer<T>::passThrough(const Matrix<T>& in){
        Matrix<T> out(1,this->m_.getM());
        for (size_t i = 0; i < this->m_.getM(); i++) { // Цикл прохода по сети
            out[0][i] = D_Neyron::FunkActiv(this->m_[0][i].Summator(in), this->F_);
        }
        return out;
    }

    template <typename T>
    void DenceLayer<T>::getWeightsFromFile(const std::string& file_name){
        getWeightsFromFile(this->m_, file_name);
    }

    template<typename T>
    void DenceLayer<T>::saveWeightsToFile(const std::string &file_name) {
        saveWeightsToFile(this->m_, file_name);
    }

    template<typename T>
    void DenceLayer<T>::BackPropagation(const DenceLayer<T> &y) {
        BackPropagation(this->m_, )
    }

    template<typename T>
    void DenceLayer<T>::BackPropagation(const Matrix<T> &y) {

    }

    template<typename T>
    void DenceLayer<T>::GradDes(Grad<T> &G, const Matrix<T> &in) {

    }


}

#endif //ARTIFICIALNN_DENCELAYER_H
