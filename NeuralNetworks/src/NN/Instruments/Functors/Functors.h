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

#define D_Sigm Sigm<double>
#define F_Sigm Sigm<float>
#define I_Sigm Sigm<int>

#define D_SigmD SigmD<double>
#define F_SigmD SigmD<float>
#define I_SigmD SigmD<int>

#define D_Relu Relu<double>
#define F_Relu Relu<float>
#define I_Relu Relu<int>

#define D_ReluD ReluD<double>
#define F_ReluD ReluD<float>
#define I_ReluD ReluD<int>

#define D_BinaryClassificator BinaryClassificator<double>
#define F_BinaryClassificator BinaryClassificator<float>
#define I_BinaryClassificator BinaryClassificator<int>

#define D_BinaryClassificatorD BinaryClassificatorD<double>
#define F_BinaryClassificatorD BinaryClassificatorD<float>
#define I_BinaryClassificatorD BinaryClassificatorD<int>
}
#endif //ARTIFICIALNN_FUNCTORS_H