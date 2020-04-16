#pragma once

namespace ANN {
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
}