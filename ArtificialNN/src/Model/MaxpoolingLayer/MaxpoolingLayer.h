#ifndef ARTIFICIALNN_MAXPOOLINGLAYER_H
#define ARTIFICIALNN_MAXPOOLINGLAYER_H

#include "Filters.h"

namespace ANN {
    
    template<typename T>
    class MaxpoolingLayer{
    public:
    MaxpoolingLayer(size_t n, size_t m);

    MaxpoolingLayer(const MaxpoolingLayer& copy);

    Matrix<T> passThrough(const Matrix<T>& in);
    Tensor<T> passThrough(const Tensor<T>& in);
    Tensor<T> BackPropagation(const Tensor<T>& input, const Tensor<T>& output,
                                       const Tensor<T> &error);

    ~MaxpoolingLayer()= default;

    private:
        size_t n_, m_;
    };

    template<typename T>
    MaxpoolingLayer<T>::MaxpoolingLayer(size_t n, size_t m) {
        n_ = n;
        m_ = m;
    }

    template<typename T>
    MaxpoolingLayer<T>::MaxpoolingLayer(const MaxpoolingLayer &copy) {
        n_ = copy.n_;
        m_ = copy.m_;
    }

    template<typename T>
    Matrix<T> MaxpoolingLayer<T>::passThrough(const Matrix<T>& in) {
        return Filter<T>::Pooling(in, n_, m_);
    }

    template<typename T>
    Tensor <T> MaxpoolingLayer<T>::passThrough(const Tensor <T> &in) {
        Tensor<T> result(in.getHeight()/this->n_, in.getWidth()/this->m_, in.getDepth());

        for(size_t i = 0; i < result.getDepth(); i++){
            result[i] = MaxpoolingLayer<T>::passThrough(in[i]);
        }
        return result;
    }

    template<typename T>
    Tensor<T>
    MaxpoolingLayer<T>::BackPropagation(const Tensor<T>& input, const Tensor<T>& output,
                                        const Tensor<T> &error) {
        return ANN::BackPropagation(input, output, error, n_, m_);
    }

}
#endif //ARTIFICIALNN_MAXPOOLINGLAYER_H
