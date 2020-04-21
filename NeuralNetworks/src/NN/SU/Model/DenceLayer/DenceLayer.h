#ifndef ARTIFICIALNN_DENCELAYER_H
#define ARTIFICIALNN_DENCELAYER_H

#include "Neyron.h"
#include "LearnNeyron.h"
#include "Initializers.h"
#include "Data.h"
#include "Layer.h"
#include <string>

namespace NN {
using std::shared_ptr;
    template<typename T>
    class DenceLayer : public Matrix<Neyron<T> >, public Layer<T>{
    public:
        DenceLayer(size_t number_neyrons, size_t number_input, shared_ptr<const Func<T>> F, shared_ptr<const Func<T>> FD,
                   shared_ptr<const Init<T>> I, double dropout_rate = 0);
        DenceLayer(const DenceLayer& copy);


        Tensor<T> passThrough(const Tensor<T>& in);
        Tensor<T> BackPropagation(const Tensor<T>& error, const Tensor<T>& in);
        void GradDes(ImpulsGrad<T>& G, const Tensor<T>& in);
        void saveToFile(std::ofstream& file);
        void getFromFile(std::ifstream& file);


        size_t getNumberImput()const{ return this->arr[0][0].getM();};

        void SimpleLearning(const Matrix<T>& a, const Matrix<T>& y, const Matrix<T>& in, double speed);

        ~DenceLayer()= default;

#ifdef TEST_DenceLayer
    public:
        const Func<T>* F_;
        const Func<T>* FD_;
        const Init<T>* I_;
        double dropout;
        Matrix<T> derivative;
        Matrix<Neyron<T> > history;
        void setZero();
#else
    private:
        shared_ptr<const Func<T>> F_;
        shared_ptr<const Func<T>> FD_;
        shared_ptr<const Init<T>> I_;
        double dropout;
        Matrix<T> derivative;
        Matrix<Neyron<T> > history;
        void setZero();
#endif
    };

#define D_DenceLayer DenceLayer<double>
#define F_DenceLayer DenceLayer<float>
#define I_DenceLayer DenceLayer<int>

    template <typename T>
    DenceLayer<T>::DenceLayer(size_t number_neyrons, size_t number_input, shared_ptr<const Func<T>> F, shared_ptr<const Func<T>> FD,
            shared_ptr<const Init<T>> I, double dropout_rate) : Matrix<Neyron<T> >(1, number_neyrons), Layer<T>("DenceLayer"){
        this->F_ = F;
        this->FD_= FD;
        this->I_ = I;
        std::cout << F_.use_count() << " ";
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
        this->dropout = copy.dropout;
    }

    template <typename T>
    Tensor<T> DenceLayer<T>::passThrough(const Tensor<T>& in){
        Tensor<T> out(1,this->m,1);
        T sum;
        for (size_t i = 0; i < this->m; i++) { // Цикл прохода по сети
            sum = this->arr[0][i].Summator(in[0]);
            out[0][0][i] = Neyron<T>::FunkActiv(sum, *(this->F_));
            this->derivative[0][i] = (*(this->FD_))(sum);
        }
        return out;
    }

    template<typename T>
    void DenceLayer<T>::SimpleLearning(const Matrix<T> &a, const Matrix<T> &y, const Matrix<T> &in, double speed) {
        if((a.getN() != y.getN())||(a.getM() != y.getM())){
            throw LearningExeption("Несовпадение размеров матрицы выхода и матрицы ожидаемых выходов!");
        }
        for(size_t i = 0; i < a.getN(); i++){
            for(size_t j = 0; j < a.getM(); j++){
                SimpleLearning(a[i][j], y[i][j],this->arr[i][j], in, speed);
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
    void DenceLayer<T>::GradDes(ImpulsGrad<T> &G, const Tensor<T> &in) {
        ::NN::GradDes(G, *this, in[0], this->history, dropout);
        setZero();
    }

    template<typename T>
    Tensor<T> DenceLayer<T>::BackPropagation(const Tensor<T> &error, const Tensor<T> &in) {
        ::NN::BackPropagation(*this, error[0], this->derivative);

        Tensor<T> result(1, getNumberImput(),1);
        result.Fill(0);
        for(size_t j = 0; j < this->getM(); j++) {
            for (size_t i = 0; i < result.getWidth(); i++) {
                result[0][0][i] += (*this)[0][j].GetD() * (*this)[0][j][0][i];
            }
        }

        return result;
    }

    template<typename T>
    void DenceLayer<T>::saveToFile(std::ofstream &file) {
        file << *this << endl << history << endl << derivative;
    }

    template<typename T>
    void DenceLayer<T>::getFromFile(std::ifstream &file) {
        file >> *this;
        file >> history;
        file >> derivative;
    }

}

#endif //ARTIFICIALNN_DENCELAYER_H
