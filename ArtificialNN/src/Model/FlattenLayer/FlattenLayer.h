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

        Tensor<T> passThrough(const Tensor<T>& in);
        Tensor<T> BackPropagation(const Tensor<T>& error, const Tensor <T>& in);
        void GradDes(const ImpulsGrad<T>& G, const Tensor <T>& in){};
        void saveToFile(const std::string& file_name);
        void getFromFile(const std::string& file_name);

        ~FlattenLayer();
    private:
        size_t height,
        width,
        depth;
    };

    template<typename T>
    Tensor<T> FlattenLayer<T>::passThrough(const Tensor<T> &in) {

        Tensor<T> m_(1, in.getDepth()*in.getHeight()*in.getWidth(), 1);

        for(size_t z = 0; z < in.getDepth(); z++){
            for(size_t x = 0; x < in.getHeight(); x++){
                for(size_t y = 0; y < in.getWidth(); y++){
                    m_[0][0][z*x*y + x*y + y] = in[z][x][y];
                }
            }
        }
        return m_[0];
    }


    template<typename T>
    Tensor<T> FlattenLayer<T>::BackPropagation(const Tensor<T>& error, const Tensor <T>& in) {
        Tensor<T> m_(height, width, depth);

        for(size_t z = 0; z < m_.getDepth(); z++){
            for(size_t x = 0; x < m_.getHeight(); x++){
                for(size_t y = 0; y < m_.getWidth(); y++){
                    m_[z][x][y] = in[0][0][z*x*y + x*y + y] ;
                }
            }
        }

        return m_;
    }

    template<typename T>
    FlattenLayer<T>::FlattenLayer(const FlattenLayer<T> &copy) : Layer<T>("FlattenLayer") {
        height = copy.height;
        width = copy.width;
        depth = copy.depth;
    }

    //TODO: write
    template<typename T>
    void FlattenLayer<T>::saveToFile(const std::string &file_name) {

    }

    //TODO: write
    template<typename T>
    void FlattenLayer<T>::getFromFile(const std::string &file_name) {

    }


    template<typename T>
    FlattenLayer<T>::~FlattenLayer() = default;
}

#endif //ARTIFICIALNN_FLATTENLAYER_H
