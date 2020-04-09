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
    class DenceLayer : public Matrix<Neyron<T> >, public Layer<T>{
    public:
        DenceLayer(size_t number_neyrons, size_t number_input, const Func<T>& F, const Func<T>& FD,
                const Init<T>& I, double dropout_rate = 0);
        DenceLayer(const DenceLayer& copy);

        Matrix<T> passThrough(const Matrix<T>& in);

        void getWeightsFromFile(const std::string& file_name);
        void saveWeightsToFile(const std::string& file_name);

        void BackPropagation(const Tensor<T>& error, const Tensor<T>& input);

        size_t getNumberImput()const{ return this->arr[0][0].getM();};
        /* TODO: Tests! */
        Matrix<T> BackPropagation() const;

        void SimpleLearning(const Matrix<T>& a, const Matrix<T>& y, const Matrix<T>& in, double speed);

        void GradDes(ImpulsGrad<T>& G, const Matrix <T>& in);

        ~DenceLayer()= default;

    private:
        const Func<T>* F_;
        const Func<T>* FD_;
        const Init<T>* I_;
        double dropout;
        Matrix<T> derivative;
        Matrix<Neyron<T> > history;
        void setZero();
    };

    template <typename T>
    DenceLayer<T>::DenceLayer(size_t number_neyrons, size_t number_input, const Func<T>& F, const Func<T>& FD,
            const Init<T>& I, double dropout_rate) : Matrix<Neyron<T> >(1, number_neyrons), Layer<T>("DenceLayer"){
        this->F_ = &F;
        this->FD_= &FD;
        this->I_ = &I;
        this->derivative = Matrix<T>(1, number_neyrons);
        this->history = Matrix<Neyron<T> >(1, number_neyrons);
        this->dropout = dropout_rate;

        for (size_t i = 0; i < number_neyrons; i++) {

            this->arr[0][i] = Neyron<T>(1, number_input);
            this->history[0][i] = Neyron<T>(1, number_input);
            this->history[0][i].GetWBias() = 0;
            this->history[0][i].Fill(0);

            for (size_t j = 0; j < number_input; j++) {
                this->arr[0][i][0][j] = (*(this->I_))();
            }
            this->arr[0][i].GetWBias() = 0;
        }
    }

    template <typename T>
    DenceLayer<T>::DenceLayer(const DenceLayer& copy):Matrix<Neyron<T> >(copy), Layer<T>(copy){
        this->F_ = copy.F_;
        this->FD_ = copy.FD_;
        this->I_ = copy.I_;
        this->derivative = copy.derivative;
        this->history = copy.history;
    }

    template <typename T>
    Matrix<T> DenceLayer<T>::passThrough(const Matrix<T>& in){
        Matrix<T> out(1,this->m);
        T sum;
        for (size_t i = 0; i < this->m; i++) { // Цикл прохода по сети
            sum = this->arr[0][i].Summator(in);
            out[0][i] = Neyron<T>::FunkActiv(sum, *(this->F_));
            this->derivative[0][i] = (*(this->FD_))(sum);
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
    void BackPropagation(const Tensor<T>& error, const Tensor<T>& input){

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

    template<typename T>
    Matrix<T> DenceLayer<T>::BackPropagation() const {
        Matrix<T> out((*this)[0][0].getN(), (*this)[0][0].getM());
        for(size_t i = 0; i < this->n; i++){
            for(size_t j = 0; j < this->m; j++){
                for(size_t x = 0; x < (*this)[i][j].getN(); x++){
                    for(size_t y = 0; y < (*this)[i][j].getM(); y++){
                        out[x][y] = (*this)[i][j].GetD() * (*this)[i][j][x][y];
                    }
                }
            }
        }
        return out;
    }

    template<typename T>
    void DenceLayer<T>::GradDes(ImpulsGrad<T> &G, const Matrix<T> &in) {
        ANN::GradDes(G, *this, in, this->history, dropout);
        setZero();
    }

}

#endif //ARTIFICIALNN_DENCELAYER_H
