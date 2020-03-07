#ifndef ARTIFICIALNN_GRAD_H
#define ARTIFICIALNN_GRAD_H

#include "Neyrons.h"

namespace ANN {

    template<typename T>
    class Grad {
    public:
        Grad() {};

        virtual void operator()(Neyron <T> &w, const Matrix <T> &in, Func <T> &F) = 0;

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
}
#endif //ARTIFICIALNN_GRAD_H
