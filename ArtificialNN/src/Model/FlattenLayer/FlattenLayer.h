#ifndef ARTIFICIALNN_FLATTENLAYER_H
#define ARTIFICIALNN_FLATTENLAYER_H

#include "Layer.h"
#include "DenceLayers.h"
#include "Layer.h"

namespace ANN {

    template<typename T>
    class FlattenLayer : public Layer<T>{
    public:
        FlattenLayer(size_t height_, size_t width_, size_t depth_) : height(height_),
            width(width_), depth(depth_), Layer<T>("FlattenLayer"){};

        FlattenLayer(const FlattenLayer<T>& copy);

        Matrix<T> passThrough(const Tensor<T>& in);

        Tensor<T> passBack(const Matrix<T>& in);
        Tensor<T> passBack(const DenceLayer<T>& in);

        ~FlattenLayer();
    private:
        size_t height,
        width,
        depth;
    };

    template<typename T>
    Matrix<T> FlattenLayer<T>::passThrough(const Tensor<T> &in) {

        Matrix<T> m_(1, in.getDepth()*in.getHeight()*in.getWidth());

        for(size_t z = 0; z < in.getDepth(); z++){
            for(size_t x = 0; x < in.getHeight(); x++){
                for(size_t y = 0; y < in.getWidth(); y++){
                    m_[0][z*x*y + x*y + y] = in[z][x][y];
                }
            }
        }
        return m_;
    }


    template<typename T>
    Tensor<T> FlattenLayer<T>::passBack(const Matrix<T> &in) {
        Tensor<T> m_(height, width, depth);

        for(size_t z = 0; z < m_.getDepth(); z++){
            for(size_t x = 0; x < m_.getHeight(); x++){
                for(size_t y = 0; y < m_.getWidth(); y++){
                    m_[z][x][y] = in[0][z*x*y + x*y + y] ;
                }
            }
        }

        return m_;
    }

    template<typename T>
    Tensor<T> FlattenLayer<T>::passBack(const DenceLayer<T> &in) {

        Matrix<T> error = in.BackPropagation();

        return FlattenLayer<T>::passBack(error, height, width, depth);
    }

    template<typename T>
    FlattenLayer<T>::FlattenLayer(const FlattenLayer<T> &copy) : Layer<T>("FlattenLayer") {
        height = copy.height;
        width = copy.width;
        depth = copy.depth;
    }


    template<typename T>
    FlattenLayer<T>::~FlattenLayer() = default;
}

#endif //ARTIFICIALNN_FLATTENLAYER_H
