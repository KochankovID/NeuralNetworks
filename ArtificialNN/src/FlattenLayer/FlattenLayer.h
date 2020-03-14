#ifndef ARTIFICIALNN_FLATTENLAYER_H
#define ARTIFICIALNN_FLATTENLAYER_H

#include "Layer.h"

namespace ANN {

    template<typename T>
    class FlattenLayer : Layer<T> {
    public:
        FlattenLayer():Layer<T>(){};

        Matrix<T> passThrough(const Matrix<T> &in);

        void BackPropagation(const Layer<T>& y);
        void BackPropagation(const Matrix<T>& y);

        void GradDes(Grad<T>& G, const Matrix <T>& in);

        ~FlattenLayer();
    private:
    };

    template<typename T>
    FlattenLayer<T>::~FlattenLayer() = default;

    template<typename T>
    Matrix<T> FlattenLayer<T>::passThrough(const Matrix<T> &in) {
        return Matrix<T>();
    }

    template<typename T>
    void FlattenLayer<T>::BackPropagation(const Layer<T> &y) {

    }

    template<typename T>
    void FlattenLayer<T>::BackPropagation(const Matrix<T> &y) {

    }

    template<typename T>
    void FlattenLayer<T>::GradDes(Grad<T> &G, const Matrix<T> &in) {
        return;
    }
}

#endif //ARTIFICIALNN_FLATTENLAYER_H
