#ifndef ARTIFICIALNN_FUNCTORS_H
#define ARTIFICIALNN_FUNCTORS_H
#include "Func.h"
#include <algorithm>
#include <math.h>

namespace NN {
// функтор
// Сигмоида
    template<typename T>
    class Sigm : public Func_speed<T> {
    public:
        explicit Sigm(const double &a_ = 1) : Func_speed<T>(a_) {};

        T operator()(const T &x) const {
            double f = 1;
            f = exp((double) -x * this->a);
            f++;
            return 1 / f;
        }

        ~Sigm() {};
    };

// Производная сигмоиды
    template<typename T>
    class SigmD : public Sigm<T> {
    public:
        SigmD(const double &a_ = 1) : Sigm<T>(a_) {};

        T operator()(const T &x) const {
            double f = 1;
            f = Sigm<T>::operator()(x) * (1 - Sigm<T>::operator()(x));
            return f;
        }

        ~SigmD() {};
    };

// Релу
    template<typename T>
    class Relu : public Func_speed<T> {
    public:
        explicit Relu(const double &a_ = 1) : Func_speed<T>(a_) {};

        T operator()(const T &x) const{
            return std::max(double(0), x * this->a);
        }

        ~Relu() {};
    };

    template<typename T>
    class ReluD : public Relu<T> {
    public:
        ReluD(const double &a_ = 1) : Relu<T>(a_) {srand(time(0));};

        T operator()(const T &x) const{
            if (x <= 0) {
                return 0;
            } else {
                return this->a;
            }
        }

        ~ReluD() {};
    };

    template<typename T>
    class BinaryClassificator : public Func<T> {
    public:
        explicit BinaryClassificator() : Func<T>() {};

        T operator()(const T &x) const {
            if(x >=0){
                return 1;
            }else{
                return 0;
            }
        }

        ~BinaryClassificator() {};
    };

    template<typename T>
    class BinaryClassificatorD : public Func<T> {
    public:
        explicit BinaryClassificatorD() : Func<T>() {};

        T operator()(const T &x) const {
            return 0;
        }

        ~BinaryClassificatorD() {};
    };

}
#endif //ARTIFICIALNN_FUNCTORS_H