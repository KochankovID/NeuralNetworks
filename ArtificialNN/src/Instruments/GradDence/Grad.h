#ifndef ARTIFICIALNN_GRAD_H
#define ARTIFICIALNN_GRAD_H

#include "Neyron.h"
#include "Filter.h"

namespace ANN {

    template<typename T>
    class ImpulsGrad {
    public:
        ImpulsGrad() {};

        virtual void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history) = 0;
        virtual void operator()(Filter<T> &F, const Matrix<T> &error, size_t step, Tensor<T>& history) = 0;

        virtual ~ImpulsGrad() {};
    private:
    };

    template<typename T>
    class ImpulsGrad_speed : public ImpulsGrad<T> {
    public:
        explicit ImpulsGrad_speed(double a_) : a(a_), ImpulsGrad<T>() {};

        virtual ~ImpulsGrad_speed() {};
    protected:
        double a;
    };

    template<typename T>
    class ImpulsGrad_speed_bordered : public ImpulsGrad_speed<T> {
    public:
        explicit ImpulsGrad_speed_bordered(double a_, double y_) : a(a_), y(y_), ImpulsGrad_speed<T>(a_) {};

        virtual ~ImpulsGrad_speed_bordered() {};
    protected:
        double a;
        double y;

        T clamps(T x){
            if(x > y){
                return  y;
            }
            if(x < -y){
                return  -y;
            }}
    };
}
#endif //ARTIFICIALNN_GRAD_H
