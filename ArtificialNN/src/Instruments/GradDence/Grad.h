#ifndef ARTIFICIALNN_GRAD_H
#define ARTIFICIALNN_GRAD_H

#include "Neyron.h"
#include "Filter.h"

namespace ANN {

    template<typename T>
    class Grad {
    public:
        Grad() {};

        virtual void operator()(Neyron <T> &w, const Matrix <T> &in) = 0;
        virtual void operator()(const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F, size_t step) = 0;


        virtual ~Grad() {};
    };

    template<typename T>
    class Grad_speed : public Grad<T> {
    public:
        explicit Grad_speed(double a_) : a(a_), Grad<T>() {};

        virtual ~Grad_speed() {};
    protected:
        double a;
    };

    template<typename T>
    class ImpulsGrad {
    public:
        ImpulsGrad() {};

        virtual void operator()(Neyron <T> &w, const Matrix <T> &in, Neyron<T>& history) = 0;
        virtual void operator()(const Matrix<T> &X, const Matrix<T> &D, Filter<T> &F,
                size_t step, Matrix<T>& history) = 0;


        virtual ~ImpulsGrad() {};
    };

    template<typename T>
    class ImpulsGrad_speed : public ImpulsGrad<T> {
    public:
        explicit ImpulsGrad_speed(double a_) : a(a_), ImpulsGrad<T>() {};

        virtual ~ImpulsGrad_speed() {};
    protected:
        double a;
    };
}
#endif //ARTIFICIALNN_GRAD_H
