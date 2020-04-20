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
        virtual void operator()(Neyron <T> &w, const Matrix<T>& in, Neyron<T>& history) const = 0;
        virtual void operator()(const Tensor<T>& in, Filter<T> &F, const Matrix<T> &error, size_t step, Filter<T>& history) const = 0;

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
}
#endif //ARTIFICIALNN_GRAD_H
