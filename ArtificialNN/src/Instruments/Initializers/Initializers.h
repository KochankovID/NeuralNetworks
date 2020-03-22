#ifndef ARTIFICIALNN_INITIALIZERS_H
#define ARTIFICIALNN_INITIALIZERS_H

#include "Init.h"
#include <random>

namespace ANN {

    template<typename T>
    class SimpleInitializator : public Init<T> {
    public:
        explicit SimpleInitializator(double k) : k_(k), Init<T>() {srand(time(0));};

        T operator()() const {
            return double((rand()) / RAND_MAX - 0.5) * k_;
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
            return double(rand()) / RAND_MAX * k_;
        }

        ~SimpleInitializatorPositive() {};
    private:
        double k_;
    };

    template<typename T>
    class allOne : public Init<T> {
    public:
        explicit allOne(double k) : k_(k), Init<T>() {};

        T operator()() const {
            srand(time(0));
            return k_;
        }

        ~allOne() {};
    private:
        double k_;
    };
}

#endif //ARTIFICIALNN_INITIALIZERS_H
