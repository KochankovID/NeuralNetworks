#ifndef ARTIFICIALNN_FUNCTORS_H
#define ARTIFICIALNN_FUNCTORS_H
#include "Func.h"
#include <algorithm>
#include <math.h>

#define D_Func Func<double, double>
#define F_Func Func<float, float>
#define I_Func Func<int, int>

// функтор
// Сигмоида
template <typename T>
class Sigm : public Func<T,T>
{
public:
    Sigm(const double& a_) : Func<T,T>(), a(a_) {};
    double a;
    T operator()(const double& x) {
        double f = 1;
        f = exp((double) -x);
        f++;
        return 1 / f;
    }
    ~Sigm() {};
};

// Производная сигмоиды
template <typename T>
class SigmD : public Sigm<T>
{
public:
    SigmD(const double& a_) : Sigm<T>(a_) {};
    T operator()(const double& x) {
        double f = 1;
        f = Sigm<T>::operator()(x)*(1 - Sigm<T>::operator()(x));
        return f;
    }
    ~SigmD() {};
};

// Релу
template <typename T>
class Relu : public Func<T,T>
{
public:
    Relu(const double& a_) : Func<T,T>(), a(a_) {};
    double a;
    T operator()(const double& x) {
        return std::max(double(0), x*a);
    }
    ~Relu() {};
};

template <typename T>
class ReluD : public Relu<T>
{
public:
    ReluD(const double& a_) : Relu<T>(a_) {};
    T a;
    T operator()(const double& x) {
        if(x < 0){
            return 0;
        }else{
            return a;
        }
    }
    ~ReluD() {};
};

#endif //ARTIFICIALNN_FUNCTORS_H