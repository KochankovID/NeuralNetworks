#ifndef ARTIFICIALNN_LAYER_H
#define ARTIFICIALNN_LAYER_H

#include "Gradients.h"
#include "Matrix.h"

namespace ANN {

    template<typename T>
    class Layer {
        Layer() = default;

        virtual Matrix<T> passThrough(const Matrix<T> &in) = 0;

        virtual void BackPropagation(const Layer<T>& y) = 0;
        virtual void BackPropagation(const Matrix<T>& y) = 0;

        virtual void GradDes(Grad<T>& G, const Matrix <T>& in) = 0;

        ~Layer() = default;
    };
};
#endif //ARTIFICIALNN_LAYER_H
