#pragma once

namespace NN {
    template<typename T>
    class Func {
    public:
        Func() {};

        virtual T operator()(const T &x) const = 0;

        virtual ~Func() {};
    };

    template<typename T>
    class Func_speed : public Func<T> {
    public:
        explicit Func_speed(double a_) : a(a_), Func<T>() {};

        virtual ~Func_speed() {};
    protected:
        double a;
    };

#define D_Func Func<double>
#define F_Func Func<float>
#define I_Func Func<int>

#define D_Func_speed Func_speed<double>
#define F_Func_speed Func_speed<float>
#define I_Func_speed Func_speed<int>
}