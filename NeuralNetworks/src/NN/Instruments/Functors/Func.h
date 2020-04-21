#pragma once

namespace NN {
    // Абстрактный класс функции
    template<typename T>
    class Func {
    public:
        // Конструкторы ---------------------------------
        // Конструктор по умолчанию
        Func() {};

        // Перегрузки операторов ------------------------
        virtual T operator()(const T &x) const = 0;

        // Деструктор -----------------------------------
        virtual ~Func() {};
    };

    // Абстрактный класс параметризированной функции
    template<typename T>
    class Func_speed : public Func<T> {
    public:
        // Конструкторы ---------------------------------
        // Конструктор инициализатор
        explicit Func_speed(double a_) : a(a_), Func<T>() {};

        // Деструктор -----------------------------------
        virtual ~Func_speed() {};
    protected:
        double a;
    };
}