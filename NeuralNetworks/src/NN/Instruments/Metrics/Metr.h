#ifndef ARTIFICIALNN_METR_H
#define ARTIFICIALNN_METR_H

#include "Matrix.h"

namespace NN {

    template<typename T>
    class Metr {
    public:
        Metr(const std::string& m_name) : m_name_(m_name) {};

        virtual Matrix<double> operator()(const Matrix<T>& out, const Matrix<T>& correct) const = 0;
        std::string getName() const { return m_name_; };

        virtual ~Metr() {};
    protected:
        std::string m_name_;
    };
}

#endif //ARTIFICIALNN_METR_H
