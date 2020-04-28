#ifndef ARTIFICIALNN_METR_H
#define ARTIFICIALNN_METR_H

#include "Matrix.h"

namespace NN {

    // Абстрактный класс метрик
    template<typename T>
    class Metr {
    public:
        // Конструкторы ---------------------------------
        Metr(const std::string& m_name) : m_name_(m_name) {};

        // Методы класса --------------------------------
        std::string getName() const { return m_name_; };

        // Перегрузки операторов ------------------------
        virtual Matrix<double> operator()(const Matrix<T>& out, const Matrix<T>& correct) const = 0;

        // Деструктор -----------------------------------
        virtual ~Metr() {};
    protected:
        // Поля класса ----------------------------------
        std::string m_name_;
    };

#define D_Metr Metr<double>
#define F_Metr Metr<float>
#define I_Metr Metr<int>
}

#endif //ARTIFICIALNN_METR_H
