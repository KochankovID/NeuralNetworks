#ifndef ARTIFICIALNN_INITIALIZERS_H
#define ARTIFICIALNN_INITIALIZERS_H

#include "Init.h"
#include <random>

namespace NN {

    template <typename T>
    T fRand(T fMin, T fMax)
    {
        T f = (double)rand() / RAND_MAX;
        return fMin + f * (fMax - fMin);
    }

    template<typename T>
    class SimpleInitializator : public Init<T> {
    public:
        explicit SimpleInitializator(double k = 1) : k_(k), Init<T>() {srand(time(0));};

        T operator()() const {
            return (double(rand()) / RAND_MAX - 0.5) * k_;
        }

        ~SimpleInitializator() {};
    private:
        double k_;
    };

    template<typename T>
    class SimpleInitializatorPositive : public Init<T> {
    public:
        explicit SimpleInitializatorPositive(double k) : k_(k), Init<T>() {srand(time(0));};

        T operator()() const {
            return (double(rand()) / RAND_MAX) * k_;
        }

        ~SimpleInitializatorPositive() {};
    private:
        double k_;
    };

    template<typename T>
    class glorot_uniform : public Init<T> {
    public:
        explicit glorot_uniform(double fan_in, double fan_out) : fan_in_(fan_in), fan_out_(fan_out),
        Init<T>() {srand(time(0));};

        T operator()() const {
            T limit = sqrt((double)6/(fan_in_+fan_out_));
            return fRand(-limit, limit);
        }

        ~glorot_uniform() {};
    private:
        double fan_out_, fan_in_;
    };
}

#endif //ARTIFICIALNN_INITIALIZERS_H
