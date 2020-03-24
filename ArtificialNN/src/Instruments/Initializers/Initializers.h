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
    class XavierInitializer : public Init<T> {
    public:
        explicit XavierInitializer(double n, double m) : m_(m), n_(n), Init<T>() {srand(time(0));};

        T operator()() const {
            return (double(rand()) / RAND_MAX) * 4/(m_+n_) -2/(m_+n_);
        }

        ~XavierInitializer() {};
    private:
        double n_, m_;
    };
}

#endif //ARTIFICIALNN_INITIALIZERS_H
