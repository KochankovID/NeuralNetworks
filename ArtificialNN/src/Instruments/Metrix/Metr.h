#ifndef ARTIFICIALNN_METR_H
#define ARTIFICIALNN_METR_H

namespace ANN {

    template<typename T>
    class Metr {
    public:
        Metr() {};

        virtual Matrix<T> operator()(const Matrix<T>& out, const Matrix<T>& correct) = 0;

        virtual ~Metr() {};
    };
}

#endif //ARTIFICIALNN_METR_H
