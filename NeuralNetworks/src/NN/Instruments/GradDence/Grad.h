#ifndef ARTIFICIALNN_GRAD_H
#define ARTIFICIALNN_GRAD_H

#include "Neyron.h"
#include "Filter.h"

namespace NN {

    template<typename T>
    class ImpulsGrad {
    public:
        ImpulsGrad(std::string type) : type_(type) {};

        std::string getType() const { return type_;};
        virtual void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history) = 0;
        virtual void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error,
                size_t step, Filter<T>& history) = 0;
        virtual void endOfExample(){};

        virtual ~ImpulsGrad() {};
    private:
        std::string type_;
    };

    template<typename T>
    class ImpulsGrad_speed : public ImpulsGrad<T> {
    public:
        explicit ImpulsGrad_speed(double a_, std::string type) : a(a_), ImpulsGrad<T>(type) {};

        virtual ~ImpulsGrad_speed() {};
    protected:
        double a;
    };

    template<typename T>
    class ImpulsGrad_speed_bordered : public ImpulsGrad_speed<T> {
    public:
        explicit ImpulsGrad_speed_bordered(double a_, double p_, std::string type)
        : a(a_), p(p_), ImpulsGrad_speed<T>(a_, type) {};

        virtual ~ImpulsGrad_speed_bordered() {};
    protected:
        double a;
        double p;

        void calculateError(const Tensor<T> &X, const Matrix<T> &error, Filter<T> &F, size_t step) const;
        void calculateError(Neyron<T>& neyron, const Matrix<T>& in) const;

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

        Matrix<T> new_D = PrepForStepM(error, step);
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
    void ImpulsGrad_speed_bordered<T>::calculateError(Neyron<T> &neyron, const Matrix<T> &in) const {
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
