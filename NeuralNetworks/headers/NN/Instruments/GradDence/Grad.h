#ifndef ARTIFICIALNN_GRAD_H
#define ARTIFICIALNN_GRAD_H

#include "Neuron.h"
#include "Filter.h"

namespace NN {

    // Абстрактный класс импульсного градиента
    template<typename T>
    class ImpulsGrad {
    public:
        // Конструкторы ---------------------------------
        ImpulsGrad(std::string type) : type_(type) {};

        // Методы класса --------------------------------
        std::string getType() const { return type_;};
        virtual void endOfExample(){};

        // Перегрузки операторов ------------------------
        virtual void operator()(Neuron <T> &w, const Matrix<T>& in, Neuron<T>& history) = 0;
        virtual void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error,
                size_t step, Filter<T>& history) = 0;

        // Деструктор -----------------------------------
        virtual ~ImpulsGrad() {};
    private:
        // Поля класса ----------------------------------
        std::string type_;
    };

    // Абстрактный класс градиентного спуска с параметром скорости обучения
    template<typename T>
    class ImpulsGrad_speed : public ImpulsGrad<T> {
    public:
        // Конструкторы ---------------------------------
        explicit ImpulsGrad_speed(double a_, std::string type) : a(a_), ImpulsGrad<T>(type) {};

        // Деструктор -----------------------------------
        virtual ~ImpulsGrad_speed() {};
    protected:
        // Поля класса ----------------------------------
        double a;
    };

    // Абстрактный класс градиентного спуска с параметром ограничения импульса
    template<typename T>
    class ImpulsGrad_speed_bordered : public ImpulsGrad_speed<T> {
    public:
        // Конструкторы ---------------------------------
        explicit ImpulsGrad_speed_bordered(double a_, double p_, std::string type)
        : a(a_), p(p_), ImpulsGrad_speed<T>(a_, type) {};

        // Деструктор -----------------------------------
        virtual ~ImpulsGrad_speed_bordered() {};
    protected:
        // Поля класса ----------------------------------
        double a;
        double p;

        // Скрытые матоды класса ------------------------
        void calculateError(const Tensor<T> &X, const Matrix<T> &error, Filter<T> &F, size_t step) const;
        void calculateError(Neuron<T>& neyron, const Matrix<T>& in) const;
        T clamps(T x) const {
            if (x > p) {
                return p;
            }
            if (x < -p) {
                return -p;
            }
            return x;
        }

    };


    template<typename T>
    void ImpulsGrad_speed_bordered<T>::calculateError(const Tensor<T> &X, const Matrix<T> &error, Filter<T> &F, size_t step) const {

        Matrix<T> new_D = _PrepForStepM(error, step);
        Tensor<T> temp(F.getHeight(), F.getWidth(), F.getDepth());

        for(size_t i = 0; i < F.getDepth(); i++){

            auto delta = Filter<T>::Svertka(X[i],new_D,1);
            if((delta.getN() != F.getHeight())||(delta.getM() != F.getWidth())){
                throw std::logic_error("Матрицы фильтра и матрицы ошибки не совпадают!");
            }
            temp[i] = delta;
        }

        F.setError(temp);
    }

    template<typename T>
    void ImpulsGrad_speed_bordered<T>::calculateError(Neuron<T> &neyron, const Matrix<T> &in) const {
        if((in.getN() != neyron.getN())||(in.getM() != neyron.getM())){
            throw std::runtime_error("Size of input matrix and neyron matrix is not equal!");
        }
        Weights<T> temp(neyron.getN(), neyron.getM());
        for (int i = 0; i < neyron.getN(); i++) {
            for (int j = 0; j < neyron.getM(); j++) {
                temp[i][j] = neyron.GetD() * in[i][j];
            }
        }
        temp.GetWBias() = neyron.GetD();
        neyron.setError(temp);
    }
}
#endif //ARTIFICIALNN_GRAD_H
