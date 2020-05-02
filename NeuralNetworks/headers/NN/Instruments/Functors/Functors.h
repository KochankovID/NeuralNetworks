#ifndef ARTIFICIALNN_FUNCTORS_H
#define ARTIFICIALNN_FUNCTORS_H

#include "Func.h"
#include <algorithm>
#include <math.h>

namespace NN {
    // Сигмоида
    template<typename T>
    class Sigm : public Func_speed<T> {
    public:
        // Конструкторы ---------------------------------
        explicit Sigm(const double &a_ = 1) : Func_speed<T>(a_) {};

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            double f = 1;
            f = exp((double) -x * this->a);
            f++;
            return 1 / f;
        }

        // Деструктор -----------------------------------
        ~Sigm() {};
    };

    // Производная сигмоиды
    template<typename T>
    class SigmD : public Sigm<T> {
    public:
        // Конструкторы ---------------------------------
        SigmD(const double &a_ = 1) : Sigm<T>(a_) {};

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            double f = 1;
            f = Sigm<T>::operator()(x) * (1 - Sigm<T>::operator()(x));
            return f;
        }

        // Деструктор -----------------------------------
        ~SigmD() {};
    };

    // Релу
    template<typename T>
    class Relu : public Func_speed<T> {
    public:
        // Конструкторы ---------------------------------
        explicit Relu(const double &a_ = 1) : Func_speed<T>(a_) {};

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            return std::max(double(0), x * this->a);
        }

        // Деструктор -----------------------------------
        ~Relu() {};
    };

    // Производная Релу
    template<typename T>
    class ReluD : public Relu<T> {
    public:
        // Конструкторы ---------------------------------
        ReluD(const double &a_ = 1) : Relu<T>(a_) { srand(time(0)); };

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            if (x <= 0) {
                return 0;
            } else {
                return this->a;
            }
        }

        // Деструктор -----------------------------------
        ~ReluD() {};
    };

    // Гиперболический тангенс
    template<typename T>
    class th : public Func_speed<T> {
    public:
        // Конструкторы ---------------------------------
        explicit th(const double &a_ = 1) : Func_speed<T>(a_) {};

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            double arg = this->a_ * x;
            arg = exp(arg);
            return (arg-1/arg)/(arg+1/arg);
        }

        // Деструктор -----------------------------------
        ~th() {};
    };

    // Производная гиперболического тангенса
    template<typename T>
    class thD : public th<T> {
    public:
        // Конструкторы ---------------------------------
        thD(const double &a_ = 1) : th<T>(a_) { srand(time(0)); };

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            return 1 - th<T>::operator()(this->a_*x)*th<T>::operator()(this->a_*x);
        }

        // Деструктор -----------------------------------
        ~thD() {};
    };

    // Гиперболический тангенс
    template<typename T>
    class LeakyRelu : public Func_speed<T> {
    public:
        // Конструкторы ---------------------------------
        explicit LeakyRelu(const double &a_ = 1) : Func_speed<T>(a_) {};

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            if (x <= 0) {
                return 0.01 * x;
            } else {
                return this->a * x;
            }
        }

        // Деструктор -----------------------------------
        ~LeakyRelu() {};
    };

    // Производная гиперболического тангенса
    template<typename T>
    class LeakyReluD : public th<T> {
    public:
        // Конструкторы ---------------------------------
        LeakyReluD(const double &a_ = 1) : th<T>(a_) { srand(time(0)); };

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            if (x <= 0) {
                return 0.01;
            } else {
                return this->a;
            }
        }

        // Деструктор -----------------------------------
        ~LeakyReluD() {};
    };

    // Бинарный классификатор
    template<typename T>
    class BinaryClassificator : public Func<T> {
    public:
        // Конструкторы ---------------------------------
        explicit BinaryClassificator() : Func<T>() {};

        // Перегрузки операторов ------------------------
        T operator()(const T &x) const {
            if (x >= 0) {
                return 1;
            } else {
                return 0;
            }
        }

        // Деструктор -----------------------------------
        ~BinaryClassificator() {};
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
}
#endif //ARTIFICIALNN_FUNCTORS_H